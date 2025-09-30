# ML Project — Income Classification (Adult Dataset)

This repository contains a complete, reproducible workflow to predict **income category** (`<=50K` vs `>50K`) from demographic and work-related attributes using multiple linear classifiers. The project is implemented in a Jupyter notebook (`ML_Project1.ipynb`) and covers custom implementations of **Perceptron** and **Adaline (GD)** as well as scikit‑learn baselines (**Perceptron**, **Adaline**, **Logistic Regression**, and **SVM**).

> **Dataset**: The project uses a two CSV datasets, (`project_adult.csv`) for training and a separate validation input file (`project_validation_inputs.csv`) for generating final predictions. These datasets are an extraction from the 1994 Census database.

---

## Contents

- `ML_Project1.ipynb` — main notebook with end‑to‑end workflow
- `Group_11_Perceptron_PredictedOutputs.csv` — predictions for the validation set (Perceptron)
- `Group_11_Adaline_PredictedOutputs.csv` — predictions for the validation set (Adaline)
- `Group_11_LogisticRegression_PredictedOutputs.csv` — predictions for the validation set (Logistic Regression)
- `Group_11_SVM_PredictedOutputs.csv` — predictions for the validation set (SVM)

> The notebook also produces diagnostic plots (e.g., **Adaline MSE vs. Epoch**) and prints training metrics during execution.

---

## Project Goals

1. Design and implement **machine learning** models using Python.
   - Develop machine learning **algorithms** for practical applications
   - Understand and implement **perceptron** and **Adaline** algorithms.
   - Apply **logistic regression** and **support vector machines** using scikit-learn.
3. **Evaluate** model performance and decision boundaries.
4. Work with real-world datasets and apply feature preprocessing.
5. **Communicate** machine learning principles and methods to diverse audiences.

---

## Data

- **Training data**: `project_adult.csv`  
  Typical columns include: `age`, `education`, `hours_per_week`, plus other demographic/work fields, with the label `income`.
- **Validation inputs**: `project_validation_inputs.csv` (no `income` column).

Target:
- `income` — binary target indicating whether the individual’s income is `<=50K` or `>50K`.

---

## Preprocessing

The notebook performs:
- **Basic cleaning** via a helper (`basic_clean`): trims, normalizes strings (e.g., replace spaces and hyphens with underscores), and checks for invalid values.
- **Type safety** utilities: `has_nan_or_inf`, `ensure_samples_by_features`.
- **Standardization** of numeric features using `StandardScaler`.
- Categorical handling: one‑hot/dummy encoding inside the notebook (and/or via model pipelines), ensuring the training/validation matrices align.

---

## Models

### Custom Implementations
- **Perceptron (custom)**: online update when a sample is misclassified (threshold at 0). Evaluated by **training accuracy** and validation predictions.
- **Adaline (GD)**: batch gradient descent on **mean squared error** loss. Tracked **MSE per epoch** and selected the **best epoch** by minimum training MSE.

Key training hyperparameters found in the notebook:
- `N_EPOCHS = 30`
- `THRESH = 0.5` (for converting Adaline raw scores to class labels)
- Reproducible shuffles via a fixed random seed

### scikit‑learn Baselines
- **Perceptron** — linear classifier using perceptron criterion
- **LogisticRegression** — linear log‑loss classifier
- **SVM (linear)** — linear support vector classifier

> Where applicable, models are wrapped in scikit‑learn **Pipelines** with standardization.

---

## Metrics & Plots

- **Classification**: `accuracy_score` on training (and optionally hold‑out splits).
- **Regression proxy (Adaline)**: `mean_squared_error` on raw outputs per epoch.
- **Visualization**: Adaline **TRAIN MSE vs. epoch** line plot to illustrate convergence and to pick the best epoch.

> The notebook prints messages like:  
> `Training accuracy at best epoch: ...`  
> `Best Adaline epoch by TRAIN MSE: ... (MSE=...)`

Replace the ellipses with your specific numbers after running the notebook.

---

## How to Run

1. **Clone** this repository and ensure Python 3.10+.
2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If you don’t have a `requirements.txt`, see the list below.
3. **Launch** Jupyter and open the notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
4. **Run all cells**. This will:
   - Load and clean the data
   - Fit the models
   - Plot diagnostics
   - Export CSVs of predictions for the validation inputs

### Expected Outputs
- `Group_11_Perceptron_PredictedOutputs.csv`
- `Group_11_Adaline_PredictedOutputs.csv`
- `Group_11_LogisticRegression_PredictedOutputs.csv`
- `Group_11_SVM_PredictedOutputs.csv`

Each CSV contains a single column with the predicted income category per validation row.

---

## Example: Reproducing Predictions (CLI)

If you extract the model‑training and prediction code into a script (optional), you might run something like:

```bash
python train_and_predict.py   --train_csv project_adult.csv   --valid_csv project_validation_inputs.csv   --model perceptron   --out Group_11_Perceptron_PredictedOutputs.csv
```

> The notebook already performs these steps; the script example is provided for a future refactor.

---

## Requirements

_Core packages used in the notebook:_
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Optional (if added later):
- `jupyterlab` or `notebook`
- `seaborn` (for enhanced visuals)

You can generate a starter `requirements.txt` with:
```bash
pip freeze | grep -E "numpy|pandas|matplotlib|scikit-learn" > requirements.txt
```

---

## Results (Fill After Running)

| Model                 | Metric            | Score  |
|----------------------:|-------------------|-------:|
| Custom Perceptron     | Accuracy (train)  | _…_    |
| Custom Adaline (best epoch) | MSE (train) | _…_    |
| sklearn Perceptron    | Accuracy (train)  | _…_    |
| Logistic Regression   | Accuracy (train)  | _…_    |
| Linear SVM            | Accuracy (train)  | _…_    |

Include any confusion matrices or ROC curves you compute as images/screenshots if desired.

---

## Notes & Design Choices

- **Why compare custom vs. sklearn?** To understand optimization behavior and convergence diagnostics (especially MSE for Adaline) and to benchmark against mature implementations.
- **Why standardize?** Linear margin‑based models often benefit from standardized features for stable optimization and consistent learning rates across features.
- **Why MSE for Adaline?** Adaline is historically trained via least‑squares; monitoring MSE reveals convergence trends even though the final task is classification.

