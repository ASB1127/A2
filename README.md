# A2

Bank Marketing term-deposit classification assignment comparing Gradient Boosted Decision Trees (GBDT) and Multi-Layer Perceptrons (MLP).

## Repository Contents

- `A2.py`: main implementation and experiment script
- `A2.ipynb`: notebook version of the same workflow
- `A2.pdf`: report
- `requirements.txt`: Python dependencies needed to run the assignment

## Requirements

- Python 3.11+ recommended
- Internet access is required the first time the script runs because the dataset is fetched from the UCI repository through `ucimlrepo`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .env
source .env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Instructions

Run the full experiment script:

```bash
python A2.py
```

What this script does:

- downloads the UCI Bank Marketing dataset
- cleans the data and adds engineered features
- creates train, validation, and test splits
- trains baseline GBDT and MLP models
- runs hyperparameter tuning for both models
- tunes the decision threshold for both models
- generates figures and summary tables used in the analysis
- prints validation and final test metrics to the terminal

Note: the full run can take a while because both models use grid search.

## Notebook

If you prefer the notebook version, open:

```bash
A2.ipynb
```

The notebook mirrors the logic in `A2.py`. You can run it in Jupyter or VS Code after installing a notebook environment.

## Expected Outputs

After running `A2.py`, the main generated artifacts include:

- GBDT training vs validation loss
- GBDT learning-rate comparison
- GBDT feature importance
- MLP training loss
- MLP depth/width comparison
- final GBDT vs MLP comparison table

## Reproducibility

- The code uses fixed random seeds where applicable, primarily `random_state=42`
- Results should be close across runs, though training time and some fitted-model details can vary slightly by machine and library version
