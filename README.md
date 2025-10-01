# ML Project — Income Classification (Adult Dataset)
---

## Overview

This repository contains a complete, reproducible workflow to predict **income category** (`<=50K` vs `>50K`) from demographic and work-related attributes using multiple linear classifiers. The project is implemented in a Jupyter notebook (`ML_Project1.ipynb`) and covers custom implementations of **Perceptron** and **Adaline (GD)** as well as scikit‑learn baselines (**Perceptron**, **Adaline**, **Logistic Regression**, and **SVM**).

> **Dataset**: The project uses a two CSV datasets - (`project_adult.csv`) for training and (`project_validation_inputs.csv`) for validation/testing. These datasets are an extraction from the 1994 Census database.

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
  Typical columns include: `age`, `education`, `hours_per_week`, plus other demographic/work fields, including an `income` column.
- **Testing data (validation inputs)**: `project_validation_inputs.csv`
  Typical columns include: `age`, `education`, `hours_per_week`, plus other demographic/work fields, **NOT** including an `income` column.

Target Variable:
- `income` — binary target indicating whether the individual’s income is `<=50K` or `>50K`.

---

## Preprocessing

The notebook performs:
- **Basic cleaning** via `basic_clean()`: trims whitespace, normalizes text (e.g., replace spaces/hyphens with underscores), and checks for invalid values
- **Type safety utilities:** `has_nan_or_inf`, `ensure_samples_by_features`
- **Standardization** of numeric features using `StandardScaler`
- **Categorical handling**: one‑hot encoding ensuring the training and validation matrices align

---

## Models

### Custom Implementations
**Perceptron (custom)**:
- Determines the number of adjustments made for misclassified sample per epoch
- Hyperparameters:
- `N_EPOCHS = 30`
- `eta = 0.1`
- `THRESH = 0.0`

**Adaline (book code)**
- Calculates the avg squared difference between the model's predicted output and the actual labels per epoch
- Hyperparameters:
- `N_EPOCHS = 30`
- `ETA_ADAL = 0.01`
- `THRESH = 0.5` (for converting Adaline raw scores to class labels)

**Scikit-learn Baselines**

**Perceptron (skearn)**
— Linear classifier using perceptron criterion

**Adaline (GD; Scikit-Learn)**
- Gradient descent-based linear model 

**LogisticRegression**
— Linear log‑loss classifier
- Outputs probability estimates
- Hyperparameters:
- `MAV_ITERS = 10000`
- `C_GRID = 0.01, 1.0, 100.0`
- `penalty = l2`
  
**SVM (linear)**
- Linear support vector classifier
- Hyperparameters:
- `MAV_ITERS = 10000`
- `C_GRID = 0.01, 1.0, 100.0`
- `penalty = l2`

> Where applicable, models are wrapped in scikit‑learn **Pipelines** with standardization.

---

## How to Run

1. **Clone** this repository and ensure Python 3.10+.
2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, see below for key packages.
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

Each CSV contains one column with the predicted income category per validation row.

---

## Requirements

_Core packages used in the notebook:_
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can generate a starter `requirements.txt` with:
```bash
pip freeze | grep -E "numpy|pandas|matplotlib|scikit-learn" > requirements.txt
```

---

## Results (Fill After Running)

| Model                 | Metric            | Score  |
|----------------------:|-------------------|-------:|
| Custom Perceptron (best epoch)| Accuracy (train)| 82.82% |
| Custom Adaline (best epoch) | MSE (train) | 78.33% |
| sklearn Perceptron    | Accuracy (train)  | 81.98% |
| sklearn Adaline       | Accuracy (train)  | 84.04% |
| Logistic Regression (best C)  | Accuracy (train)  | 85.18%|
| Linear SVM          (best C)  | Accuracy (train)  | 85.19%|




