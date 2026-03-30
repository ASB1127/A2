import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace(list(MISSING_STRING_VALUES), pd.NA)

    df[TARGET_COL] = (
        df[TARGET_COL]
        .str.lower()
        .map(TARGET_MAP)
        .astype("Int64")
    )

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
