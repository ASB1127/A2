# Changes Log

## Project Direction

- Chose the **Bank Marketing** dataset from UCI.
- Chose a **`.py` script with `# %%` notebook-style cells** instead of a Jupyter notebook so the code can still be run block-by-block.
- Kept the implementation **CPU-only**.

## Data Loading

- Added `load_data()` in [A2.py](/Users/amitbal/A2/A2.py) to fetch the dataset from `ucimlrepo`.
- Combined `bank_marketing.data.features` and `bank_marketing.data.targets` into a single pandas `DataFrame`.

## Schema Definition

- Defined the raw numeric columns as:
  - `age`
  - `balance`
  - `day_of_week`
  - `duration`
  - `campaign`
  - `pdays`
  - `previous`
- Defined the raw categorical columns as:
  - `job`
  - `marital`
  - `education`
  - `default`
  - `housing`
  - `loan`
  - `contact`
  - `month`
  - `poutcome`
- Defined the target column as `y`.
- Added `TARGET_MAP = {"yes": 1, "no": 0}`.

## Data Cleaning and Validation

- Added `validate_data()` in [A2.py](/Users/amitbal/A2/A2.py).
- Stripped whitespace from column names.
- Converted numeric columns with `pd.to_numeric(..., errors="coerce")`.
- Kept categorical columns as strings/objects for downstream preprocessing.
- Standardized missing markers with:
  - `""`
  - `"?"`
  - `"nan"`
  - `"none"`
  - `"null"`
- Replaced those markers with `np.nan`.
- Mapped the binary target from `yes/no` to `1/0`.
- Adjusted the cleaning logic to avoid pandas `pd.NA` issues inside sklearn imputers.

## Feature Engineering

- Added `add_engineered_features()` in [A2.py](/Users/amitbal/A2/A2.py).
- Added `previously_contacted`:
  - `df["previously_contacted"] = (df["pdays"] >= 0).astype(int)`
  - Purpose: separate the prior-contact status encoded by the `pdays` sentinel from the raw recency value.
- Added `age_group`:
  - bucketed `age` into `18-30`, `31-45`, `46-60`, and `60+`
  - Purpose: provide a more interpretable age representation and allow the models to use broader age patterns.
- Added `campaign_ratio_previous`:
  - `df["campaign_ratio_previous"] = df["campaign"] / (df["previous"] + 1)`
  - Purpose: capture current campaign intensity relative to prior outreach history.

## Pipeline Verification Blocks

- Added notebook-style execution blocks in [A2.py](/Users/amitbal/A2/A2.py) to inspect:
  - raw data preview
  - validated data preview
  - column dtypes
  - missing values per column
  - target distribution
  - engineered features preview
- Verified the pipeline outputs by running the script.
- Confirmed:
  - target mapping works
  - missing values are present mainly in `poutcome`, `contact`, `education`, and `job`
  - engineered features are being created as expected
  - class distribution is imbalanced but valid

## Split and Preprocessing

- Added `split_data()` in [A2.py](/Users/amitbal/A2/A2.py) for a stratified `70/15/15` split.
- Used the same engineered dataset split for both GBDT and MLP.
- Added `get_feature_types()` to avoid duplicated logic when separating numeric and categorical modeling features.
- Added `build_preprocessor()` for train-only preprocessing.

### GBDT Preprocessing

- Numeric columns:
  - median imputation
- Categorical columns:
  - impute true missing values with `"MISSING"`
  - one-hot encode using `OneHotEncoder(handle_unknown="ignore")`
- No numeric scaling applied

### MLP Preprocessing

- Numeric columns:
  - median imputation
  - `StandardScaler()`
- Categorical columns:
  - impute true missing values with `"MISSING"`
  - one-hot encode using `OneHotEncoder(handle_unknown="ignore")`

- Added `preprocess_data()` to bundle:
  - cleaning
  - feature engineering
  - split creation
  - train-only preprocessing
  - transformed matrices for both GBDT and MLP

## Baseline GBDT

- Added baseline `XGBClassifier` setup in [A2.py](/Users/amitbal/A2/A2.py) with:
  - `objective="binary:logistic"`
  - `eval_metric="logloss"`
  - `n_estimators=300`
  - `learning_rate=0.1`
  - `max_depth=6`
  - `subsample=1.0`
  - `colsample_bytree=1.0`
  - `random_state=42`
- Added a separate training block for the baseline GBDT.
- Recorded total training time.

## Baseline GBDT Evaluation

- Added training accuracy calculation for an overfitting check.
- Added validation metrics:
  - accuracy
  - precision
  - recall
  - F1-score
  - AUC-PR via `average_precision_score`

## GBDT Hyperparameter Tuning

- Added `tune_gbdt_hyperparameters()` in [A2.py](/Users/amitbal/A2/A2.py) so the tuning logic lives in a separate function instead of being embedded directly in a long notebook block.
- Removed `use_label_encoder=False` from XGBoost model definitions because the installed XGBoost version does not use that parameter and emits warnings.
- Current tuning grid includes:
  - `learning_rate`: `0.01`, `0.1`, `0.3`
  - `n_estimators`: `100`, `300`
  - `max_depth`: `3`, `6`
  - `subsample`: `0.8`, `1.0`
  - `reg_alpha`: `0.0`
  - `reg_lambda`: `1.0`, `5.0`
- Each run:
  - fits on the training set
  - evaluates on the validation set
  - stores validation metrics and training time
- Added logic to keep the best tuned model using:
  - highest validation `f1`
  - `auc_pr` as the tiebreaker
- Added a separate notebook-style call block that:
  - runs `tune_gbdt_hyperparameters(...)`
  - prints the top 5 validation runs
  - prints the best tuned result

## GBDT Threshold Tuning

- Added `tune_decision_threshold()` in [A2.py](/Users/amitbal/A2/A2.py) to evaluate multiple probability cutoffs on the validation set after model training.
- Current threshold candidates are:
  - `0.3`
  - `0.4`
  - `0.5`
  - `0.6`
  - `0.7`
- The helper:
  - converts validation probabilities into class predictions at each threshold
  - computes accuracy, precision, recall, F1, and AUC-PR
  - selects the best threshold by validation `f1`, with `auc_pr` as a tiebreaker
- Added a separate notebook-style block that:
  - gets validation probabilities from the best tuned GBDT model
  - runs threshold tuning
  - prints the full threshold comparison table
  - prints the best threshold result

## Baseline MLP

- Added `MLPClassifier` import in [A2.py](/Users/amitbal/A2/A2.py).
- Added a separate notebook-style baseline MLP definition block with:
  - `hidden_layer_sizes=(128, 64)`
  - `activation="relu"`
  - `learning_rate_init=0.001`
  - `alpha=0.0001`
  - `early_stopping=True`
  - `max_iter=600`
  - `random_state=42`
- Kept the MLP implementation in scikit-learn to match the assignment requirement to use `sklearn.neural_network.MLPClassifier`.
- Added a separate notebook-style training block for the baseline MLP.
- Recorded total MLP training time.
- Added MLP training accuracy calculation for an overfitting check.
- Added baseline MLP validation metrics:
  - accuracy
  - precision
  - recall
  - F1-score
  - AUC-PR via `average_precision_score`

## MLP Hyperparameter Tuning

- Added `tune_mlp_hyperparameters()` in [A2.py](/Users/amitbal/A2/A2.py) so the MLP search logic is kept in a separate function.
- Current MLP tuning grid includes:
  - `hidden_layer_sizes`: `(64,)`, `(128,)`, `(128, 64)`
  - `activation`: `"relu"`, `"tanh"`
  - `learning_rate_init`: `0.001`, `0.01`
  - `alpha`: `0.0001`, `0.001`, `0.01`
  - `max_iter`: `600`
- Enabled `early_stopping=True` for tuned MLP runs to reduce overfitting and stop training when validation performance stops improving.
- Shifted the tuning search toward smaller architectures after the larger network showed signs of overfitting.
- Each run:
  - fits on the MLP-preprocessed training set
  - evaluates on the validation set
  - stores validation metrics and training time
- Added logic to keep the best tuned MLP model using:
  - highest validation `f1`
  - `auc_pr` as the tiebreaker
- Added a separate notebook-style call block that:
  - runs `tune_mlp_hyperparameters(...)`
  - prints the top 5 validation runs
  - prints the best tuned MLP result
- Added a notebook-style matplotlib block to plot the tuned MLP training loss curve using `mlp_best_model.loss_curve_`.
- Added a notebook-style matplotlib block to visualize the effect of network depth/width on validation performance by plotting the best validation F1 achieved by each `hidden_layer_sizes` setting.
- Saved the MLP visualizations to the `vis/` directory as:
  - `vis/mlp_training_loss_curve.png`
  - `vis/mlp_depth_width_validation_f1.png`

## Final Model Selection and Test Evaluation

- Added `evaluate_model_on_test_set()` in [A2.py](/Users/amitbal/A2/A2.py) to evaluate a chosen trained model on the test set with a specified decision threshold.
- Added `build_comparison_table()` in [A2.py](/Users/amitbal/A2/A2.py) to assemble the final side-by-side results table.
- Added `plot_comparison_table()` in [A2.py](/Users/amitbal/A2/A2.py) to render the final comparison table as a matplotlib figure for the report.
- Added `VIS_DIR = Path("vis")` in [A2.py](/Users/amitbal/A2/A2.py) and configured visualizations to save into that directory.
- Ordered the final comparison table columns as:
  - model
  - accuracy
  - precision
  - recall
  - F1-score
  - AUC-PR
  - training time
- Rounded the final comparison table values for cleaner report presentation.
- Added a separate notebook-style block to choose the final models:
  - final GBDT model: `gbdt_best_model`
  - final GBDT threshold: best threshold found on the validation set
  - final MLP model: `mlp_best_model`
  - final MLP threshold: `0.5`
- Added a separate notebook-style block to evaluate both final models on the test set exactly once.
- Added a separate notebook-style block to print the final GBDT vs MLP comparison table including:
  - accuracy
  - precision
  - recall
  - F1-score
  - AUC-PR
  - training time
- Updated the final comparison block to display the side-by-side table as a matplotlib visualization instead of only printing a raw DataFrame.
- Saved the final comparison visualization as:
  - `vis/final_comparison_table.png`

## Key Design Decisions to Mention in the Report

- All learned preprocessing is fit on the training split only.
- `unknown` is treated as a meaningful category unless it appears as a true missing marker.
- The dataset is imbalanced, so evaluation emphasizes precision, recall, F1, and AUC-PR rather than accuracy alone.
- Feature engineering was kept limited to a few interpretable additions so the GBDT vs MLP comparison stays clear.
