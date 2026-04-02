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
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    cross_val_score,
    train_test_split,
    validation_curve,
)
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
def build_model_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool,
    model,
) -> Pipeline:
    """Wrap preprocessing and an estimator into one leakage-safe pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", model),
        ]
    )


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


def tune_gbdt_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """Tune GBDT hyperparameters on the train/validation split."""
    param_grid = {
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 6],
        "model__subsample": [0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1, 0.5],
        "model__reg_lambda": [0.5, 1.0, 5.0],
    }

    base_pipeline = build_model_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=False,
        model=XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            colsample_bytree=1.0,
            random_state=42,
        ),
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    total_search_time = time.time() - start_time

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[
        [
            "param_model__learning_rate",
            "param_model__n_estimators",
            "param_model__max_depth",
            "param_model__subsample",
            "param_model__reg_alpha",
            "param_model__reg_lambda",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].rename(
        columns={
            "param_model__learning_rate": "learning_rate",
            "param_model__n_estimators": "n_estimators",
            "param_model__max_depth": "max_depth",
            "param_model__subsample": "subsample",
            "param_model__reg_alpha": "reg_alpha",
            "param_model__reg_lambda": "reg_lambda",
            "mean_test_score": "cv_f1",
            "std_test_score": "cv_f1_std",
        }
    )

    best_pipeline = grid_search.best_estimator_
    best_params = {
        key.replace("model__", ""): value for key, value in grid_search.best_params_.items()
    }
    y_val_pred = best_pipeline.predict(X_val)
    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]

    best_metrics = {
        **best_params,
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred, zero_division=0),
        "recall": recall_score(y_val, y_val_pred, zero_division=0),
        "f1": f1_score(y_val, y_val_pred, zero_division=0),
        "auc_pr": average_precision_score(y_val, y_val_proba),
        "train_time_sec": grid_search.refit_time_,
        "search_time_sec": total_search_time,
    }

    validation_rows = []
    raw_param_grid = {
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [100, 300],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 5.0],
    }
    for params in ParameterGrid(raw_param_grid):
        pipeline = build_model_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            scale_numeric=False,
            model=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                colsample_bytree=1.0,
                random_state=42,
                **params,
            ),
        )
        train_start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - train_start_time

        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

        validation_rows.append(
            {
                **params,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, zero_division=0),
                "recall": recall_score(y_val, y_val_pred, zero_division=0),
                "f1": f1_score(y_val, y_val_pred, zero_division=0),
                "auc_pr": average_precision_score(y_val, y_val_proba),
                "train_time_sec": train_time,
            }
        )

    validation_results_df = pd.DataFrame(validation_rows)
    results_df = results_df.merge(
        validation_results_df,
        on=["learning_rate", "n_estimators", "max_depth", "subsample", "reg_alpha", "reg_lambda"],
        how="left",
    )
    results_df = results_df.sort_values(["f1", "auc_pr"], ascending=False).reset_index(drop=True)

    return results_df, best_metrics, best_pipeline


# %%
def summarize_gbdt_parameter_effects(results_df: pd.DataFrame, parameter_name: str) -> pd.DataFrame:
    """Summarize how one GBDT hyperparameter affects performance and overfitting."""
    summary_df = (
        results_df.groupby(parameter_name, as_index=False)
        .agg(
            mean_train_f1=("train_f1", "mean"),
            mean_val_f1=("f1", "mean"),
            mean_val_auc_pr=("auc_pr", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .round(4)
    )
    summary_df["overfit_gap_f1"] = (
        summary_df["mean_train_f1"] - summary_df["mean_val_f1"]
    ).round(4)
    return summary_df


# %%
def plot_gbdt_parameter_summary_table(summary_df: pd.DataFrame, parameter_name: str):
    """Render one GBDT parameter-effect summary as a matplotlib table."""
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title(f"GBDT Parameter Effect Summary: {parameter_name}", pad=12)
    plt.tight_layout()
    plt.savefig(
        VIS_DIR / f"gbdt_parameter_summary_{parameter_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


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
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
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
def tune_mlp_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """Tune MLP hyperparameters on the train/validation split."""
    param_grid = {
        "model__hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
        "model__activation": ["relu", "tanh"],
        "model__learning_rate_init": [0.001, 0.01, 0.1],
        "model__max_iter": [300, 600],
    }

    base_pipeline = build_model_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=True,
        model=MLPClassifier(
            early_stopping=True,
            random_state=42,
        ),
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    total_search_time = time.time() - start_time

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[
        [
            "param_model__hidden_layer_sizes",
            "param_model__activation",
            "param_model__learning_rate_init",
            "param_model__max_iter",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].rename(
        columns={
            "param_model__hidden_layer_sizes": "hidden_layer_sizes",
            "param_model__activation": "activation",
            "param_model__learning_rate_init": "learning_rate_init",
            "param_model__max_iter": "max_iter",
            "mean_test_score": "cv_f1",
            "std_test_score": "cv_f1_std",
        }
    )

    best_pipeline = grid_search.best_estimator_
    best_params = {
        key.replace("model__", ""): value for key, value in grid_search.best_params_.items()
    }
    y_train_pred = best_pipeline.predict(X_train)
    y_val_pred = best_pipeline.predict(X_val)
    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]

    best_metrics = {
        **best_params,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred, zero_division=0),
        "recall": recall_score(y_val, y_val_pred, zero_division=0),
        "f1": f1_score(y_val, y_val_pred, zero_division=0),
        "auc_pr": average_precision_score(y_val, y_val_proba),
        "train_time_sec": grid_search.refit_time_,
        "search_time_sec": total_search_time,
    }

    validation_rows = []
    raw_param_grid = {
        "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [300, 600],
    }
    for params in ParameterGrid(raw_param_grid):
        pipeline = build_model_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            scale_numeric=True,
            model=MLPClassifier(
                early_stopping=True,
                random_state=42,
                **params,
            ),
        )
        train_start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - train_start_time
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

        validation_rows.append(
            {
                **params,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, zero_division=0),
                "recall": recall_score(y_val, y_val_pred, zero_division=0),
                "f1": f1_score(y_val, y_val_pred, zero_division=0),
                "auc_pr": average_precision_score(y_val, y_val_proba),
                "train_time_sec": train_time,
            }
        )

    validation_results_df = pd.DataFrame(validation_rows)
    results_df = results_df.merge(
        validation_results_df,
        on=["hidden_layer_sizes", "activation", "learning_rate_init", "max_iter"],
        how="left",
    )
    results_df = results_df.sort_values(["f1", "auc_pr"], ascending=False).reset_index(drop=True)

    return results_df, best_metrics, best_pipeline


# %%
def evaluate_model_on_test_set(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate a trained model once on the test set."""
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
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
def summarize_final_model_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize which final model leads on each metric and on training time."""
    summary_rows = []
    metric_directions = {
        "accuracy": "max",
        "precision": "max",
        "recall": "max",
        "f1": "max",
        "auc_pr": "max",
        "train_time_sec": "min",
    }

    for metric_name, direction in metric_directions.items():
        if direction == "max":
            best_idx = comparison_df[metric_name].idxmax()
        else:
            best_idx = comparison_df[metric_name].idxmin()

        summary_rows.append(
            {
                "metric": metric_name,
                "best_model": comparison_df.loc[best_idx, "model"],
                "best_value": round(comparison_df.loc[best_idx, metric_name], 4),
            }
        )

    return pd.DataFrame(summary_rows)


def format_transformed_feature_name(feature_name: str, categorical_cols: list[str]) -> str:
    """Convert transformed column names into cleaner plot labels."""
    clean_name = feature_name
    for prefix in ("num__", "cat__", "remainder__", "imputer__", "scaler__", "onehot__"):
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]

    for col in sorted(categorical_cols, key=len, reverse=True):
        column_prefix = f"{col}_"
        if clean_name.startswith(column_prefix):
            return f"{col} = {clean_name[len(column_prefix):]}"

    return clean_name


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    categorical_cols: list[str],
) -> list[str]:
    """Return readable transformed feature names from a fitted preprocessor."""
    raw_feature_names = preprocessor.get_feature_names_out()
    return [
        format_transformed_feature_name(feature_name, categorical_cols)
        for feature_name in raw_feature_names
    ]


def plot_named_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    max_num_features: int = 15,
):
    """Plot GBDT feature importances using readable names and rounded labels."""
    importance_by_feature_id = model.get_booster().get_score(importance_type="gain")

    importance_rows = []
    for feature_id, importance_value in importance_by_feature_id.items():
        feature_index = int(feature_id[1:])
        importance_rows.append(
            {
                "feature": feature_names[feature_index],
                "importance": float(importance_value),
            }
        )

    importance_df = (
        pd.DataFrame(importance_rows)
        .sort_values("importance", ascending=False)
        .head(max_num_features)
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    fig_height = max(6, 0.5 * len(importance_df) + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(importance_df["feature"], importance_df["importance"], color="#2c6fb7")

    max_importance = importance_df["importance"].max()
    label_offset = max_importance * 0.01 if max_importance > 0 else 0.01

    for bar, importance_value in zip(bars, importance_df["importance"]):
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{importance_value:.2f}",
            va="center",
        )

    ax.set_title("GBDT Feature Importance")
    ax.set_xlabel("Importance score")
    ax.set_ylabel("Features")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, max_importance * 1.12 if max_importance > 0 else 1)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "gbdt_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()


def fit_gbdt_with_monitoring(X_train, y_train, X_val, y_val, params: dict, early_stopping_rounds: int = 20):
    """Fit a GBDT model with eval_set monitoring and early stopping."""
    monitoring_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        colsample_bytree=1.0,
        random_state=42,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    monitoring_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return monitoring_model


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
    "precision": precision_score(y_val, y_val_pred_gbdt, zero_division=0),
    "recall": recall_score(y_val, y_val_pred_gbdt, zero_division=0),
    "f1": f1_score(y_val, y_val_pred_gbdt, zero_division=0),
    "auc_pr": average_precision_score(y_val, y_val_proba_gbdt),
}

print(f"Baseline GBDT training accuracy: {gbdt_train_accuracy:.4f}")

print("\nBaseline GBDT validation metrics:")
for metric_name, metric_value in gbdt_val_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


# %%
gbdt_tuning_results_df, gbdt_best_metrics, gbdt_best_model = tune_gbdt_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    numeric_cols,
    categorical_cols,
)

print("Top 5 tuned GBDT runs by validation F1:")
print(gbdt_tuning_results_df.head(5))

print("\nBest tuned GBDT result:")
print(gbdt_best_metrics)


# %%
gbdt_parameters_to_summarize = [
    "learning_rate",
    "n_estimators",
    "max_depth",
    "subsample",
    "reg_alpha",
    "reg_lambda",
]

for parameter_name in gbdt_parameters_to_summarize:
    print(f"\nGBDT parameter effect summary: {parameter_name}")
    parameter_summary_df = summarize_gbdt_parameter_effects(
        gbdt_tuning_results_df,
        parameter_name,
    )
    print(parameter_summary_df)
    plot_gbdt_parameter_summary_table(parameter_summary_df, parameter_name)


# %%
gbdt_monitoring_params = {
    "learning_rate": gbdt_best_metrics["learning_rate"],
    "n_estimators": gbdt_best_metrics["n_estimators"],
    "max_depth": gbdt_best_metrics["max_depth"],
    "subsample": gbdt_best_metrics["subsample"],
    "reg_alpha": gbdt_best_metrics["reg_alpha"],
    "reg_lambda": gbdt_best_metrics["reg_lambda"],
}

gbdt_monitor_start_time = time.time()
gbdt_monitored_model = fit_gbdt_with_monitoring(
    X_train_gbdt,
    y_train,
    X_val_gbdt,
    y_val,
    gbdt_monitoring_params,
    early_stopping_rounds=20,
)
gbdt_monitored_train_time_sec = time.time() - gbdt_monitor_start_time

gbdt_eval_results = gbdt_monitored_model.evals_result()
train_logloss = gbdt_eval_results["validation_0"]["logloss"]
val_logloss = gbdt_eval_results["validation_1"]["logloss"]

plt.figure(figsize=(8, 5))
plt.plot(train_logloss, label="Train Log Loss", linewidth=2)
plt.plot(val_logloss, label="Validation Log Loss", linewidth=2)
plt.title("GBDT Training vs Validation Loss")
plt.xlabel("Boosting Round")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIS_DIR / "gbdt_train_vs_validation_loss.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
gbdt_feature_names = get_transformed_feature_names(
    gbdt_preprocessor,
    categorical_cols,
)
plot_named_feature_importance(
    gbdt_monitored_model,
    gbdt_feature_names,
    max_num_features=15,
)


# %%
learning_rate_values = [0.01, 0.1, 0.3]
learning_rate_results = []
learning_rate_loss_curves = {}

for learning_rate in learning_rate_values:
    learning_rate_model = fit_gbdt_with_monitoring(
        X_train_gbdt,
        y_train,
        X_val_gbdt,
        y_val,
        {
            "learning_rate": learning_rate,
            "n_estimators": gbdt_best_metrics["n_estimators"],
            "max_depth": gbdt_best_metrics["max_depth"],
            "subsample": gbdt_best_metrics["subsample"],
            "reg_alpha": gbdt_best_metrics["reg_alpha"],
            "reg_lambda": gbdt_best_metrics["reg_lambda"],
        },
        early_stopping_rounds=20,
    )

    y_val_pred_lr = learning_rate_model.predict(X_val_gbdt)
    y_val_proba_lr = learning_rate_model.predict_proba(X_val_gbdt)[:, 1]
    learning_rate_loss_curves[learning_rate] = learning_rate_model.evals_result()["validation_1"]["logloss"]

    learning_rate_results.append(
        {
            "learning_rate": learning_rate,
            "best_iteration": learning_rate_model.best_iteration,
            "val_f1": f1_score(y_val, y_val_pred_lr, zero_division=0),
            "val_auc_pr": average_precision_score(y_val, y_val_proba_lr),
            "min_val_logloss": min(
                learning_rate_model.evals_result()["validation_1"]["logloss"]
            ),
        }
    )

learning_rate_results_df = pd.DataFrame(learning_rate_results).sort_values("learning_rate")
print("GBDT learning rate comparison:")
print(learning_rate_results_df)

plt.figure(figsize=(9, 5))
for learning_rate in learning_rate_values:
    plt.plot(
        learning_rate_loss_curves[learning_rate],
        label=f"learning_rate = {learning_rate}",
        linewidth=2,
    )
plt.title("Validation Loss Curves Across Learning Rates")
plt.xlabel("Boosting Round")
plt.ylabel("Validation Log Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIS_DIR / "gbdt_learning_rate_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
gbdt_best_val_proba = gbdt_monitored_model.predict_proba(X_val_gbdt)[:, 1]

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
    "precision": precision_score(y_val, y_val_pred_mlp, zero_division=0),
    "recall": recall_score(y_val, y_val_pred_mlp, zero_division=0),
    "f1": f1_score(y_val, y_val_pred_mlp, zero_division=0),
    "auc_pr": average_precision_score(y_val, y_val_proba_mlp),
}

print(f"Baseline MLP training accuracy: {mlp_train_accuracy:.4f}")

print("\nBaseline MLP validation metrics:")
for metric_name, metric_value in mlp_val_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


# %%
mlp_tuning_results_df, mlp_best_metrics, mlp_best_pipeline = tune_mlp_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    numeric_cols,
    categorical_cols,
)

print("Top 5 tuned MLP runs by validation F1:")
print(mlp_tuning_results_df.head(5))

print("\nBest tuned MLP result:")
print(mlp_best_metrics)


# %%
plt.figure(figsize=(8, 5))
plt.plot(mlp_best_pipeline.named_steps["model"].loss_curve_, linewidth=2)
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
target_mlp_architecture = (128, 64)
target_mlp_architecture_results = mlp_tuning_results_df[
    mlp_tuning_results_df["hidden_layer_sizes"] == target_mlp_architecture
].sort_values(["f1", "auc_pr"], ascending=False)

if target_mlp_architecture_results.empty:
    print(f"No tuned MLP runs found for hidden_layer_sizes = {target_mlp_architecture}.")
else:
    best_target_mlp_result = target_mlp_architecture_results.iloc[0]

    print(f"Best tuned MLP run for hidden_layer_sizes = {target_mlp_architecture}:")
    print(
        best_target_mlp_result[
            [
                "hidden_layer_sizes",
                "activation",
                "learning_rate_init",
                "max_iter",
                "train_accuracy",
                "train_f1",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc_pr",
                "train_time_sec",
            ]
        ]
    )

    print(f"\nOverfitting check for best hidden_layer_sizes = {target_mlp_architecture} MLP:")
    print(f"Training accuracy: {best_target_mlp_result['train_accuracy']:.4f}")
    print(f"Validation accuracy: {best_target_mlp_result['accuracy']:.4f}")
    print(f"Training F1: {best_target_mlp_result['train_f1']:.4f}")
    print(f"Validation F1: {best_target_mlp_result['f1']:.4f}")


# %%
mlp_best_val_proba = mlp_best_pipeline.predict_proba(X_val)[:, 1]

mlp_threshold_results_df, mlp_best_threshold_result = tune_decision_threshold(
    y_val,
    mlp_best_val_proba,
    threshold_candidates,
)

print("MLP threshold tuning results:")
print(mlp_threshold_results_df)

print("\nBest MLP threshold result:")
print(mlp_best_threshold_result)


# %%
final_gbdt_model = gbdt_monitored_model
final_gbdt_threshold = gbdt_best_threshold_result["threshold"]

final_mlp_model = mlp_best_pipeline
final_mlp_threshold = mlp_best_threshold_result["threshold"]

print("Final GBDT configuration:")
print(gbdt_best_metrics)
print(f"Final GBDT threshold: {final_gbdt_threshold}")

print("\nFinal MLP configuration:")
print(mlp_best_metrics)
print(f"Final MLP threshold: {final_mlp_threshold}")


# %%
gbdt_cv_model = build_model_pipeline(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    scale_numeric=False,
    model=XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        colsample_bytree=1.0,
        random_state=42,
        learning_rate=gbdt_best_metrics["learning_rate"],
        n_estimators=gbdt_best_metrics["n_estimators"],
        max_depth=gbdt_best_metrics["max_depth"],
        subsample=gbdt_best_metrics["subsample"],
        reg_alpha=gbdt_best_metrics["reg_alpha"],
        reg_lambda=gbdt_best_metrics["reg_lambda"],
    ),
)

mlp_cv_model = build_model_pipeline(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    scale_numeric=True,
    model=MLPClassifier(
        early_stopping=True,
        random_state=42,
        hidden_layer_sizes=mlp_best_metrics["hidden_layer_sizes"],
        activation=mlp_best_metrics["activation"],
        learning_rate_init=mlp_best_metrics["learning_rate_init"],
        max_iter=mlp_best_metrics["max_iter"],
    ),
)

gbdt_cv_scores = cross_val_score(
    gbdt_cv_model,
    X_train,
    y_train,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)
mlp_cv_scores = cross_val_score(
    mlp_cv_model,
    X_train,
    y_train,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)

cv_summary_df = pd.DataFrame(
    [
        {
            "model": "GBDT",
            "cv_f1_mean": round(gbdt_cv_scores.mean(), 4),
            "cv_f1_std": round(gbdt_cv_scores.std(), 4),
        },
        {
            "model": "MLP",
            "cv_f1_mean": round(mlp_cv_scores.mean(), 4),
            "cv_f1_std": round(mlp_cv_scores.std(), 4),
        },
    ]
)

print("Cross-validation F1 summary:")
print(cv_summary_df)


# %%
# Additional diagnostic beyond the tuned search grid: sweep a wider range of
# max_depth values so the validation-curve plot is informative rather than a
# two-point comparison.
gbdt_validation_curve_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            build_preprocessor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                colsample_bytree=1.0,
                random_state=42,
                learning_rate=gbdt_best_metrics["learning_rate"],
                n_estimators=gbdt_best_metrics["n_estimators"],
                subsample=gbdt_best_metrics["subsample"],
                reg_alpha=gbdt_best_metrics["reg_alpha"],
                reg_lambda=gbdt_best_metrics["reg_lambda"],
            ),
        ),
    ]
)

depth_param_range = [2, 3, 4, 5, 6, 7, 8]
gbdt_train_scores, gbdt_val_scores = validation_curve(
    gbdt_validation_curve_pipeline,
    X_train,
    y_train,
    param_name="model__max_depth",
    param_range=depth_param_range,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)

gbdt_train_mean = gbdt_train_scores.mean(axis=1)
gbdt_val_mean = gbdt_val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(depth_param_range, gbdt_train_mean, marker="o", linewidth=2, label="Train F1")
plt.plot(depth_param_range, gbdt_val_mean, marker="o", linewidth=2, label="Validation F1")
plt.title("GBDT Validation Curve for max_depth")
plt.xlabel("max_depth")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIS_DIR / "gbdt_validation_curve_max_depth.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
final_gbdt_test_metrics = evaluate_model_on_test_set(
    final_gbdt_model,
    X_test_gbdt,
    y_test,
    threshold=final_gbdt_threshold,
)
final_gbdt_test_metrics["train_time_sec"] = gbdt_monitored_train_time_sec

final_mlp_test_metrics = evaluate_model_on_test_set(
    final_mlp_model,
    X_test,
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

final_comparison_summary = summarize_final_model_comparison(final_comparison_table)
print("\nFinal comparison summary:")
print(final_comparison_summary)
