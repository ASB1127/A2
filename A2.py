import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# %%
from ucimlrepo import fetch_ucirepo


# %%
MISSING_STRING_VALUES = {"", "?", "nan", "none", "null"}


# %%
NUMERIC_COLS = [
    "age",
    "balance",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

TARGET_COL = "y"
TARGET_MAP = {"yes": 1, "no": 0}
VIS_DIR = Path("vis")
VIS_DIR.mkdir(exist_ok=True)


# %%
def load_data() -> pd.DataFrame:
    """Fetch the Bank Marketing dataset and return one combined DataFrame."""
    bank_marketing = fetch_ucirepo(id=222)

    X = bank_marketing.data.features.copy()
    y = bank_marketing.data.targets.copy()

    df = pd.concat([X, y], axis=1)
    return df


# %%
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column types, standardize missing markers, and map the target."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("object")
        df[col] = df[col].where(~pd.isna(df[col]), np.nan)
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
        df[col] = df[col].replace(list(MISSING_STRING_VALUES), np.nan)

    df[TARGET_COL] = df[TARGET_COL].astype("object")
    df[TARGET_COL] = df[TARGET_COL].where(~pd.isna(df[TARGET_COL]), np.nan)
    df[TARGET_COL] = df[TARGET_COL].map(
        lambda value: value.strip().lower() if isinstance(value, str) else value
    )
    df[TARGET_COL] = df[TARGET_COL].replace(list(MISSING_STRING_VALUES), np.nan)
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP).astype("Int64")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features used in the comparison."""
    df = df.copy()
    df["previously_contacted"] = (df["pdays"] >= 0).astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"],
    )
    df["campaign_ratio_previous"] = df["campaign"] / (df["previous"] + 1)
    return df


# %%
def get_feature_types(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[list[str], list[str], list[str]]:
    """Return feature, numeric, and categorical column lists for modeling."""
    feature_cols = [col for col in df.columns if col != target_col]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


# %%
def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """Create stratified train, validation, and test splits."""
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    X = df.drop(columns=target_col)
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state,
    )

    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# %%
def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a train-only preprocessing pipeline for either GBDT or MLP."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


# %%
def preprocess_data(
    df: pd.DataFrame,
    use_engineered_features: bool = True,
    random_state: int = 42,
) -> dict:
    """Run the full preprocessing workflow for both GBDT and MLP."""
    cleaned_df = validate_data(df)

    if use_engineered_features:
        cleaned_df = add_engineered_features(cleaned_df)

    feature_cols, numeric_cols, categorical_cols = get_feature_types(cleaned_df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        cleaned_df,
        target_col=TARGET_COL,
        random_state=random_state,
    )

    gbdt_preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=False,
    )
    mlp_preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=True,
    )

    X_train_gbdt = gbdt_preprocessor.fit_transform(X_train)
    X_val_gbdt = gbdt_preprocessor.transform(X_val)
    X_test_gbdt = gbdt_preprocessor.transform(X_test)

    X_train_mlp = mlp_preprocessor.fit_transform(X_train)
    X_val_mlp = mlp_preprocessor.transform(X_val)
    X_test_mlp = mlp_preprocessor.transform(X_test)

    return {
        "cleaned_df": cleaned_df,
        "feature_names": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "gbdt": {
            "preprocessor": gbdt_preprocessor,
            "X_train": X_train_gbdt,
            "X_val": X_val_gbdt,
            "X_test": X_test_gbdt,
        },
        "mlp": {
            "preprocessor": mlp_preprocessor,
            "X_train": X_train_mlp,
            "X_val": X_val_mlp,
            "X_test": X_test_mlp,
        },
    }


# %%
def tune_gbdt_hyperparameters(X_train, y_train, X_val, y_val) -> tuple[pd.DataFrame, dict, XGBClassifier]:
    """Tune GBDT hyperparameters on the train/validation split."""
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [100, 300],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
        "reg_alpha": [0.0],
        "reg_lambda": [1.0, 5.0],
    }

    tuning_results = []
    best_metrics = None
    best_model = None

    for params in ParameterGrid(param_grid):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            colsample_bytree=1.0,
            random_state=42,
            **params,
        )

        start_time = time.time()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        train_time = time.time() - start_time

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        result = {
            **params,
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred),
            "recall": recall_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred),
            "auc_pr": average_precision_score(y_val, y_val_proba),
            "train_time_sec": train_time,
        }
        tuning_results.append(result)

        if (
            best_metrics is None
            or result["f1"] > best_metrics["f1"]
            or (
                result["f1"] == best_metrics["f1"]
                and result["auc_pr"] > best_metrics["auc_pr"]
            )
        ):
            best_metrics = result
            best_model = model

    results_df = (
        pd.DataFrame(tuning_results)
        .sort_values(["f1", "auc_pr"], ascending=False)
        .reset_index(drop=True)
    )

    return results_df, best_metrics, best_model


# %%
def tune_decision_threshold(
    y_true,
    y_proba,
    thresholds: list[float],
) -> tuple[pd.DataFrame, dict]:
    """Evaluate multiple probability thresholds on the validation set."""
    threshold_results = []
    best_result = None

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        result = {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc_pr": average_precision_score(y_true, y_proba),
        }
        threshold_results.append(result)

        if (
            best_result is None
            or result["f1"] > best_result["f1"]
            or (
                result["f1"] == best_result["f1"]
                and result["auc_pr"] > best_result["auc_pr"]
            )
        ):
            best_result = result

    results_df = pd.DataFrame(threshold_results).sort_values("threshold").reset_index(drop=True)
    return results_df, best_result


# %%
def tune_mlp_hyperparameters(X_train, y_train, X_val, y_val) -> tuple[pd.DataFrame, dict, MLPClassifier]:
    """Tune MLP hyperparameters on the train/validation split."""
    param_grid = {
        "hidden_layer_sizes": [(64,), (128,), (128, 64)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [600],
    }

    tuning_results = []
    best_metrics = None
    best_model = None

    for params in ParameterGrid(param_grid):
        model = MLPClassifier(
            early_stopping=True,
            random_state=42,
            **params,
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        result = {
            **params,
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred),
            "recall": recall_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred),
            "auc_pr": average_precision_score(y_val, y_val_proba),
            "train_time_sec": train_time,
        }
        tuning_results.append(result)

        if (
            best_metrics is None
            or result["f1"] > best_metrics["f1"]
            or (
                result["f1"] == best_metrics["f1"]
                and result["auc_pr"] > best_metrics["auc_pr"]
            )
        ):
            best_metrics = result
            best_model = model

    results_df = (
        pd.DataFrame(tuning_results)
        .sort_values(["f1", "auc_pr"], ascending=False)
        .reset_index(drop=True)
    )

    return results_df, best_metrics, best_model


# %%
def evaluate_model_on_test_set(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate a trained model once on the test set."""
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "auc_pr": average_precision_score(y_test, y_test_proba),
    }


# %%
def build_comparison_table(gbdt_results: dict, mlp_results: dict) -> pd.DataFrame:
    """Build a side-by-side final comparison table."""
    comparison_df = pd.DataFrame(
        [
            {"model": "GBDT", **gbdt_results},
            {"model": "MLP", **mlp_results},
        ]
    )
    column_order = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_pr",
        "train_time_sec",
    ]
    comparison_df = comparison_df[column_order].round(4)
    return comparison_df


# %%
def plot_comparison_table(comparison_df: pd.DataFrame, title: str = "GBDT vs MLP Comparison"):
    """Render the final comparison table as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=comparison_df.values,
        colLabels=comparison_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    fig.savefig(VIS_DIR / "final_comparison_table.png", dpi=300, bbox_inches="tight")
    plt.show()


# %%
loaded_df = load_data()
print("Raw data preview:")
print(loaded_df.head())

df = validate_data(loaded_df)
print("\nValidated data preview:")
print(df.head())

print("\nColumn dtypes:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))

print("\nTarget distribution:")
print(df[TARGET_COL].value_counts(dropna=False))

engineered_df = add_engineered_features(df)
print("\nEngineered data preview:")
print(engineered_df.head())

print("\nEngineered feature check:")
print(
    engineered_df[
        [
            "pdays",
            "previously_contacted",
            "age",
            "age_group",
            "campaign",
            "previous",
            "campaign_ratio_previous",
        ]
    ].head(10)
)


# %%
X_train, X_val, X_test, y_train, y_val, y_test = split_data(engineered_df)

feature_cols, numeric_cols, categorical_cols = get_feature_types(engineered_df)

gbdt_preprocessor = build_preprocessor(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    scale_numeric=False,
)
mlp_preprocessor = build_preprocessor(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    scale_numeric=True,
)

X_train_gbdt = gbdt_preprocessor.fit_transform(X_train)
X_val_gbdt = gbdt_preprocessor.transform(X_val)
X_test_gbdt = gbdt_preprocessor.transform(X_test)

X_train_mlp = mlp_preprocessor.fit_transform(X_train)
X_val_mlp = mlp_preprocessor.transform(X_val)
X_test_mlp = mlp_preprocessor.transform(X_test)

print("\nSplit sizes:")
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

print("\nTarget distribution by split:")
print("Train")
print(y_train.value_counts(normalize=True))
print("Validation")
print(y_val.value_counts(normalize=True))
print("Test")
print(y_test.value_counts(normalize=True))

print("\nTransformed feature shapes:")
print("GBDT:", X_train_gbdt.shape, X_val_gbdt.shape, X_test_gbdt.shape)
print("MLP:", X_train_mlp.shape, X_val_mlp.shape, X_test_mlp.shape)


# %%
gbdt_baseline = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
)

print(gbdt_baseline)


# %%
start_time = time.time()

gbdt_baseline.fit(
    X_train_gbdt,
    y_train,
    eval_set=[(X_train_gbdt, y_train), (X_val_gbdt, y_val)],
    verbose=False,
)

gbdt_train_time = time.time() - start_time

print(f"Baseline GBDT training time: {gbdt_train_time:.4f} seconds")


# %%
y_train_pred_gbdt = gbdt_baseline.predict(X_train_gbdt)
y_val_pred_gbdt = gbdt_baseline.predict(X_val_gbdt)
y_val_proba_gbdt = gbdt_baseline.predict_proba(X_val_gbdt)[:, 1]

gbdt_train_accuracy = accuracy_score(y_train, y_train_pred_gbdt)

gbdt_val_metrics = {
    "accuracy": accuracy_score(y_val, y_val_pred_gbdt),
    "precision": precision_score(y_val, y_val_pred_gbdt),
    "recall": recall_score(y_val, y_val_pred_gbdt),
    "f1": f1_score(y_val, y_val_pred_gbdt),
    "auc_pr": average_precision_score(y_val, y_val_proba_gbdt),
}

print(f"Baseline GBDT training accuracy: {gbdt_train_accuracy:.4f}")

print("\nBaseline GBDT validation metrics:")
for metric_name, metric_value in gbdt_val_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


# %%
gbdt_tuning_results_df, gbdt_best_metrics, gbdt_best_model = tune_gbdt_hyperparameters(
    X_train_gbdt,
    y_train,
    X_val_gbdt,
    y_val,
)

print("Top 5 tuned GBDT runs by validation F1:")
print(gbdt_tuning_results_df.head(5))

print("\nBest tuned GBDT result:")
print(gbdt_best_metrics)


# %%
threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
gbdt_best_val_proba = gbdt_best_model.predict_proba(X_val_gbdt)[:, 1]

gbdt_threshold_results_df, gbdt_best_threshold_result = tune_decision_threshold(
    y_val,
    gbdt_best_val_proba,
    threshold_candidates,
)

print("GBDT threshold tuning results:")
print(gbdt_threshold_results_df)

print("\nBest GBDT threshold result:")
print(gbdt_best_threshold_result)


# %%
mlp_baseline = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    learning_rate_init=0.001,
    alpha=0.0001,
    early_stopping=True,
    max_iter=600,
    random_state=42,
)

print(mlp_baseline)


# %%
start_time = time.time()

mlp_baseline.fit(X_train_mlp, y_train)

mlp_train_time = time.time() - start_time

print(f"Baseline MLP training time: {mlp_train_time:.4f} seconds")


# %%
y_train_pred_mlp = mlp_baseline.predict(X_train_mlp)
y_val_pred_mlp = mlp_baseline.predict(X_val_mlp)
y_val_proba_mlp = mlp_baseline.predict_proba(X_val_mlp)[:, 1]

mlp_train_accuracy = accuracy_score(y_train, y_train_pred_mlp)

mlp_val_metrics = {
    "accuracy": accuracy_score(y_val, y_val_pred_mlp),
    "precision": precision_score(y_val, y_val_pred_mlp),
    "recall": recall_score(y_val, y_val_pred_mlp),
    "f1": f1_score(y_val, y_val_pred_mlp),
    "auc_pr": average_precision_score(y_val, y_val_proba_mlp),
}

print(f"Baseline MLP training accuracy: {mlp_train_accuracy:.4f}")

print("\nBaseline MLP validation metrics:")
for metric_name, metric_value in mlp_val_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


# %%
mlp_tuning_results_df, mlp_best_metrics, mlp_best_model = tune_mlp_hyperparameters(
    X_train_mlp,
    y_train,
    X_val_mlp,
    y_val,
)

print("Top 5 tuned MLP runs by validation F1:")
print(mlp_tuning_results_df.head(5))

print("\nBest tuned MLP result:")
print(mlp_best_metrics)


# %%
plt.figure(figsize=(8, 5))
plt.plot(mlp_best_model.loss_curve_, linewidth=2)
plt.title("MLP Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIS_DIR / "mlp_training_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
depth_width_performance = (
    mlp_tuning_results_df.groupby("hidden_layer_sizes", as_index=False)["f1"]
    .max()
    .sort_values("f1", ascending=False)
)

architecture_labels = [str(architecture) for architecture in depth_width_performance["hidden_layer_sizes"]]

plt.figure(figsize=(8, 5))
plt.bar(architecture_labels, depth_width_performance["f1"])
plt.title("Effect of Network Depth/Width on Validation F1")
plt.xlabel("Hidden Layer Sizes")
plt.ylabel("Validation F1")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(VIS_DIR / "mlp_depth_width_validation_f1.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
final_gbdt_model = gbdt_best_model
final_gbdt_threshold = gbdt_best_threshold_result["threshold"]

final_mlp_model = mlp_best_model
final_mlp_threshold = 0.5

print("Final GBDT configuration:")
print(gbdt_best_metrics)
print(f"Final GBDT threshold: {final_gbdt_threshold}")

print("\nFinal MLP configuration:")
print(mlp_best_metrics)
print(f"Final MLP threshold: {final_mlp_threshold}")


# %%
final_gbdt_test_metrics = evaluate_model_on_test_set(
    final_gbdt_model,
    X_test_gbdt,
    y_test,
    threshold=final_gbdt_threshold,
)
final_gbdt_test_metrics["train_time_sec"] = gbdt_best_metrics["train_time_sec"]

final_mlp_test_metrics = evaluate_model_on_test_set(
    final_mlp_model,
    X_test_mlp,
    y_test,
    threshold=final_mlp_threshold,
)
final_mlp_test_metrics["train_time_sec"] = mlp_best_metrics["train_time_sec"]

print("Final GBDT test metrics:")
print(final_gbdt_test_metrics)

print("\nFinal MLP test metrics:")
print(final_mlp_test_metrics)


# %%
final_comparison_table = build_comparison_table(
    final_gbdt_test_metrics,
    final_mlp_test_metrics,
)

print("Final GBDT vs MLP comparison table:")
print(final_comparison_table)
plot_comparison_table(final_comparison_table)
