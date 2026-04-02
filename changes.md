# Changes Log

## Project Direction

- Chose the **Bank Marketing** dataset from UCI.
- Kept the main implementation in [A2.py](/Users/amitbal/A2/A2.py) using `# %%` notebook-style cells so the workflow can be run as either a script or a notebook-like sequence.
- Regenerated [A2.ipynb](/Users/amitbal/A2/A2.ipynb) from [A2.py](/Users/amitbal/A2/A2.py) whenever the script changed so both submission formats stayed aligned.
- Kept the implementation CPU-only.

## Data Loading

- Added `load_data()` in [A2.py](/Users/amitbal/A2/A2.py) to fetch the Bank Marketing dataset from `ucimlrepo`.
- Combined the raw feature matrix and target vector into a single pandas `DataFrame`.

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
- Kept categorical columns as string/object data for downstream preprocessing.
- Standardized missing markers:
  - `""`
  - `"?"`
  - `"nan"`
  - `"none"`
  - `"null"`
- Replaced standardized missing markers with `np.nan`.
- Mapped the binary target from `yes/no` to `1/0`.
- Adjusted cleaning to avoid pandas missing-value edge cases inside sklearn imputers.

## Feature Engineering

- Added `add_engineered_features()` in [A2.py](/Users/amitbal/A2/A2.py).
- Added `previously_contacted`:
  - `df["previously_contacted"] = (df["pdays"] >= 0).astype(int)`
  - Purpose: separate prior-contact status from the raw `pdays` recency value.
- Added `age_group`:
  - bucketed `age` into `18-30`, `31-45`, `46-60`, and `60+`
  - Purpose: capture broader age patterns with a more interpretable feature.
- Added `campaign_ratio_previous`:
  - `df["campaign_ratio_previous"] = df["campaign"] / (df["previous"] + 1)`
  - Purpose: compare current campaign intensity to prior outreach history.

## Pipeline Verification Blocks

- Added notebook-style execution blocks in [A2.py](/Users/amitbal/A2/A2.py) to inspect:
  - raw data preview
  - validated data preview
  - column dtypes
  - missing values per column
  - target distribution
  - engineered feature preview
- Verified by running the script and checking:
  - target mapping
  - missing-value locations
  - engineered feature creation
  - class imbalance across the dataset

## Split and Preprocessing

- Added `split_data()` in [A2.py](/Users/amitbal/A2/A2.py) for a stratified `70/15/15` split.
- Used the same engineered dataset split for both GBDT and MLP.
- Added `get_feature_types()` to separate numeric and categorical model inputs.
- Added `build_preprocessor()` for train-only preprocessing.
- Added `build_model_pipeline()` so preprocessing and the model can be wrapped into one sklearn `Pipeline`.
- Updated tuning and cross-validation logic to use leakage-safe pipelines on raw `X_train`, instead of cross-validating already-transformed matrices.

### GBDT Preprocessing

- Numeric columns:
  - median imputation
- Categorical columns:
  - impute true missing values with `"MISSING"`
  - one-hot encode with `OneHotEncoder(handle_unknown="ignore")`
- No numeric scaling applied

### MLP Preprocessing

- Numeric columns:
  - median imputation
  - `StandardScaler()`
- Categorical columns:
  - impute true missing values with `"MISSING"`
  - one-hot encode with `OneHotEncoder(handle_unknown="ignore")`

- Kept `preprocess_data()` in [A2.py](/Users/amitbal/A2/A2.py) to bundle:
  - cleaning
  - feature engineering
  - split creation
  - transformed matrices for baseline/final monitored fits

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
- Added baseline validation metrics:
  - accuracy
  - precision
  - recall
  - F1-score
  - AUC-PR

## GBDT Hyperparameter Tuning

- Added `tune_gbdt_hyperparameters()` in [A2.py](/Users/amitbal/A2/A2.py).
- Replaced the manual GBDT hyperparameter loop with `GridSearchCV`.
- Moved GBDT tuning onto a preprocessing+model pipeline so each CV fold refits preprocessing safely.
- Current GBDT tuning grid includes:
  - `learning_rate`: `0.01`, `0.1`, `0.3`
  - `n_estimators`: `100`, `300`
  - `max_depth`: `3`, `6`
  - `subsample`: `0.8`, `1.0`
  - `reg_alpha`: `0.0`, `0.1`, `0.5`
  - `reg_lambda`: `0.5`, `1.0`, `5.0`
- `GridSearchCV` uses 3-fold cross-validation with `scoring="f1"` and `refit=True`.
- After CV selection, the code still computes held-out validation metrics and training-time summaries for each parameter combination for reporting consistency.
- Added notebook-style blocks to:
  - print the top 5 tuned runs
  - print the best tuned result
  - summarize each GBDT hyperparameter effect
- Added `summarize_gbdt_parameter_effects()` and `plot_gbdt_parameter_summary_table()` in [A2.py](/Users/amitbal/A2/A2.py).
- Saved parameter-summary tables to `vis/` as:
  - `vis/gbdt_parameter_summary_learning_rate.png`
  - `vis/gbdt_parameter_summary_n_estimators.png`
  - `vis/gbdt_parameter_summary_max_depth.png`
  - `vis/gbdt_parameter_summary_subsample.png`
  - `vis/gbdt_parameter_summary_reg_alpha.png`
  - `vis/gbdt_parameter_summary_reg_lambda.png`
- Added `fit_gbdt_with_monitoring()` to refit the selected configuration with `eval_set` and `early_stopping_rounds`.
- Added plots for:
  - training vs validation loss
  - learning-rate comparison
  - feature importance
- Saved GBDT visualizations to:
  - `vis/gbdt_train_vs_validation_loss.png`
  - `vis/gbdt_feature_importance.png`
  - `vis/gbdt_learning_rate_comparison.png`

## GBDT Feature Importance

- Replaced the earlier generic XGBoost feature-importance view with a custom named-feature plot.
- Added:
  - `format_transformed_feature_name()`
  - `get_transformed_feature_names()`
  - `plot_named_feature_importance()`
- Mapped one-hot encoded feature IDs back to readable names such as `month = mar` and `poutcome = success`.
- Rounded plotted importance labels to two decimals.

## GBDT Threshold Tuning

- Added `tune_decision_threshold()` in [A2.py](/Users/amitbal/A2/A2.py) to evaluate multiple probability cutoffs on the validation set.
- Current threshold candidates are:
  - `0.3`
  - `0.4`
  - `0.5`
  - `0.6`
  - `0.7`
- The helper:
  - converts validation probabilities into predictions at each threshold
  - computes accuracy, precision, recall, F1, and AUC-PR
  - selects the best threshold by validation `f1`, with `auc_pr` as a tiebreaker
- Added a notebook-style block that:
  - gets validation probabilities from the monitored GBDT model
  - runs threshold tuning
  - prints the threshold comparison table
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
- Kept the MLP implementation in scikit-learn to match the assignment requirement.
- Added a separate training block for the baseline MLP.
- Recorded total MLP training time.
- Added baseline validation metrics:
  - accuracy
  - precision
  - recall
  - F1-score
  - AUC-PR

## MLP Hyperparameter Tuning

- Added `tune_mlp_hyperparameters()` in [A2.py](/Users/amitbal/A2/A2.py).
- Replaced the manual MLP hyperparameter loop with `GridSearchCV`.
- Moved MLP tuning onto a preprocessing+model pipeline so scaling and encoding are refit safely inside each CV fold.
- Current MLP tuning grid includes:
  - `hidden_layer_sizes`: `(64,)`, `(128, 64)`, `(256, 128, 64)`
  - `activation`: `"relu"`, `"tanh"`
  - `learning_rate_init`: `0.001`, `0.01`, `0.1`
  - `max_iter`: `300`, `600`
- Enabled `early_stopping=True` for tuned MLP runs.
- `GridSearchCV` uses 3-fold cross-validation with `scoring="f1"` and `refit=True`.
- After CV selection, the code still computes per-configuration training metrics, held-out validation metrics, and training time for reporting consistency.
- Added notebook-style blocks to:
  - print the top 5 tuned runs
  - print the best tuned MLP result
  - inspect the best tuned run where `hidden_layer_sizes == (128, 64)`
  - print a direct overfitting check for that architecture
- Added plots for:
  - tuned MLP training loss
  - effect of depth/width on validation F1
- Saved MLP visualizations to:
  - `vis/mlp_training_loss_curve.png`
  - `vis/mlp_depth_width_validation_f1.png`

## MLP Threshold Tuning

- Reused `tune_decision_threshold()` for MLP so the final comparison uses symmetric threshold tuning for both model families.
- Added a notebook-style block that:
  - gets validation probabilities from the final tuned MLP pipeline
  - evaluates the same threshold candidates used for GBDT
  - prints the threshold comparison table
  - prints the best threshold result
- Updated the final MLP evaluation to use the tuned threshold instead of the default `0.5`.

## Cross-Validation and Diagnostics

- Added explicit `cross_val_score` usage for the final GBDT and MLP pipelines.
- Added a 3-fold F1 cross-validation summary for both final model families.
- Added a GBDT `validation_curve` diagnostic for `max_depth` using a local preprocessing pipeline on raw `X_train`.
- Swept `max_depth` across `2` through `8` to make the validation-curve plot more informative than a two-point comparison.
- Saved the validation-curve plot to:
  - `vis/gbdt_validation_curve_max_depth.png`

## Final Model Selection and Test Evaluation

- Added `evaluate_model_on_test_set()` in [A2.py](/Users/amitbal/A2/A2.py) to evaluate a trained model on the test set with a specified threshold.
- Added `build_comparison_table()` and `plot_comparison_table()` for the final side-by-side comparison.
- Added `summarize_final_model_comparison()` to identify which model leads on each final metric.
- Final model choices are:
  - final GBDT model: early-stopped monitored GBDT fit
  - final GBDT threshold: best validation threshold from threshold tuning
  - final MLP model: tuned MLP pipeline
  - final MLP threshold: best validation threshold from threshold tuning
- Added notebook-style blocks to:
  - print final model configurations
  - evaluate both final models on the test set
  - print the final comparison table
  - print the metric-by-metric comparison summary
- Saved the final comparison visualization as:
  - `vis/final_comparison_table.png`

## Key Design Decisions to Mention in the Report

- All learned preprocessing is fit on the training split only, and CV-based tuning now uses leakage-safe preprocessing pipelines.
- The dataset is imbalanced, so evaluation emphasizes precision, recall, F1, and AUC-PR rather than accuracy alone.
- Threshold tuning is part of the final model comparison because recall/F1 tradeoffs matter for the minority class.
- Feature engineering was kept limited to a few interpretable additions so the GBDT vs MLP comparison stayed clear.
- The final reported GBDT is the early-stopped monitored model so the reported test metrics align with the `eval_set` and loss-curve workflow.
- Feature-importance values should be interpreted as predictive signals rather than causal explanations.
