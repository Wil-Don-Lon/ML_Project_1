# Machine Learning Project 1

## Overview
This project applies and compares several machine learning algorithms to the **Adult Income dataset** (`adult.csv`). The goal is to predict whether an individual’s income exceeds $50K/year based on demographic and life experience variables.

The notebook walks through preprocessing, model training, evaluation, and comparison of:
- **Perceptron**
- **Adaline (Adaptive Linear Neuron)**
- **Support Vector Machines (SVM)**
- **Logistic Regression**

This project was designed to give hands-on experience with classical machine learning algorithms and optimization techniques.

---

## Repository Structure
- `ML_Project1.ipynb` – Main Jupyter Notebook containing data loading, preprocessing, model training, and evaluation.
- `project_adult.csv` – Training dataset (Adult Income dataset).
- `project_validation_inputs.csv` – Validation dataset for model evaluation.

---

## Requirements
Install dependencies with:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### Main Libraries Used
- **pandas / numpy** – Data manipulation
- **scipy** – Scientific computations
- **scikit-learn** – ML models (Perceptron, Adaline, SVM, Logistic Regression)
- **matplotlib / seaborn** – Visualization

---

## Workflow

1. **Data Loading**  
   - Import training (`project_adult.csv`) and validation (`project_validation_inputs.csv`) datasets.
   - Explore variables (demographic and categorical attributes such as work class, education, marital status, occupation, race, and sex).

2. **Preprocessing**  
   - Apply **One-Hot Encoding** for categorical features.
   - Use **StandardScaler** to normalize numerical attributes.
   - Handle missing values and clean dataset.

3. **Model Training & Evaluation**  
   - Train and evaluate the following models:
     - **Perceptron** – Linear binary classifier.
     - **Adaline** – Linear regression-like model with gradient descent.
     - **SVM** – Classification with different kernel options.
     - **Logistic Regression** – Statistical classification approach.
   - Compare results across training and validation sets.

4. **Performance Metrics**  
   - Accuracy
   - Confusion matrices
   - Decision boundary visualizations (where applicable)

---

## Results
- **Perceptron / Adaline**: Demonstrate linear decision-making ability but limited in capturing nonlinear relationships.  
- **SVM**: Strong classification performance, especially with RBF kernel.  
- **Logistic Regression**: Balanced performance, interpretable coefficients.  

---

## How to Run
1. Clone or download this repository.  
2. Place the datasets (`project_adult.csv`, `project_validation_inputs.csv`) in the same directory as `ML_Project1.ipynb`.  
3. Open the notebook:

```bash
jupyter notebook ML_Project1.ipynb
```

4. Run all cells to reproduce preprocessing, model training, and evaluation.

---

## Notes
- Make sure dataset file names match the ones used in the notebook.
- Results may vary slightly depending on random seed initialization.
