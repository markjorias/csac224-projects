<div align="center">

# CSAC 224 · Machine Learning

**Four Jupyter-notebook problem sets: linear regression, logistic regression and a small neural network.**

![Python 3.12 tested](https://img.shields.io/badge/python-3.12%20(tested)-3776AB?logo=python&logoColor=white)
![Notebooks: 4](https://img.shields.io/badge/notebooks-4-F37626?logo=jupyter&logoColor=white)
![Period: Feb-Mar 2025](https://img.shields.io/badge/period-Feb%E2%80%93Mar%202025-blue)
![Status: coursework archive](https://img.shields.io/badge/status-coursework%20archive-lightgrey)

</div>

## Overview

This repository collects the problem sets I completed for the CSAC 224 Machine Learning course. Each problem set is a Jupyter notebook that loads a small public dataset, prepares it, trains a model and evaluates it. Every problem set also has a short PDF report and the dataset it uses, all in the same folder.

**Skills demonstrated**

- Loading and inspecting tabular data with `pandas`
- Train/test splitting and feature standardization (`StandardScaler`)
- Simple linear regression and mean squared error (`scikit-learn`)
- Encoding categorical features (`LabelEncoder`) and dimensionality reduction (`PCA`)
- Logistic regression, L1/L2 penalties, accuracy and confusion matrices
- A small feed-forward neural network in Keras (ReLU hidden layer, softmax output, Adam, early stopping)
- Plotting results with `matplotlib` and `seaborn`

## Table of contents

- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Programs](#programs)
- [Methodology](#methodology)
- [Sample results](#sample-results)

## Getting started

**Requirements**

- Python 3 with Jupyter. The notebooks were last run on Python 3.10–3.13 according to their saved metadata, and I re-ran them on Python 3.12.14.
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` and `tensorflow` (only problem set 4 needs TensorFlow).
- Optional: `pydot` and Graphviz, only for the model diagram in problem set 4. Without them that cell prints a message and the notebook carries on.

Versions used for the sample results below: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, TensorFlow/Keras 2.21.0, matplotlib 3.11.2, seaborn 0.13.2.

**Install**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow notebook
```

**Run a notebook interactively.** The notebooks read their data with relative paths, so start Jupyter from inside the problem-set folder:

```bash
cd problem-sets/01-linear-regression-chickenpox
jupyter notebook linear-regression-chickenpox.ipynb
```

**Run a notebook headless.** This writes the executed copy outside the repository:

```bash
cd problem-sets/01-linear-regression-chickenpox
jupyter nbconvert --to notebook --execute --output-dir /tmp/ml-out linear-regression-chickenpox.ipynb
```

Replace the folder and notebook name to run the others. Problem set 2 does not run as submitted: it reads `housing.csv`, which is not included in the repository.

## Repository structure

```
.
├── README.md
├── .gitignore
└── problem-sets/
    ├── 01-linear-regression-chickenpox/
    │   ├── linear-regression-chickenpox.ipynb
    │   ├── linear-regression-chickenpox.pdf
    │   ├── hungary-chickenpox.csv
    │   └── hungary-county-edges.csv
    ├── 02-linear-regression-concrete/
    │   ├── linear-regression-concrete.ipynb
    │   ├── linear-regression-concrete.pdf
    │   └── concrete-data.xls
    ├── 03-logistic-regression-mushrooms/
    │   ├── logistic-regression-mushrooms.ipynb
    │   ├── logistic-regression-mushrooms.pdf
    │   └── mushrooms.csv
    └── 04-neural-network-car-evaluation/
        ├── neural-network-car-evaluation.ipynb
        ├── neural-network-car-evaluation.pdf
        └── car-evaluation.csv
```

## Programs

**Dates are file last-modified dates** of each notebook, taken from the file system. This folder has no git history, so they may reflect when files were saved or copied rather than when the work was first written. The four PDF reports were last modified on 2025-03-16, except problem set 4's, which was last modified on 2025-03-14.

### Linear regression

| Notebook | Description | Concept practiced | Last modified |
|---|---|---|---|
| [01-linear-regression-chickenpox](problem-sets/01-linear-regression-chickenpox/linear-regression-chickenpox.ipynb) | Predicts weekly chickenpox cases in one Hungarian county (VAS) from another (BARANYA) with an 80/20 split, `LinearRegression`, train/test plots and test MSE. | Simple linear regression, train/test split, MSE | 2025-02-01 |
| [02-linear-regression-concrete](problem-sets/02-linear-regression-concrete/linear-regression-concrete.ipynb) | Same workflow as problem set 1, but the notebook as submitted reads `housing.csv`, which is not in the repo. Its PDF report and the bundled `concrete-data.xls` are about concrete compressive strength. | Simple linear regression, train/test split, MSE | 2025-02-05 |

### Classification

| Notebook | Description | Concept practiced | Last modified |
|---|---|---|---|
| [03-logistic-regression-mushrooms](problem-sets/03-logistic-regression-mushrooms/logistic-regression-mushrooms.ipynb) | Classifies mushrooms as edible or poisonous: label-encodes the 22 features, reduces them to 7 principal components, standardizes, fits logistic regression and compares it with L1 and L2 penalties (C = 0.5). | Categorical encoding, PCA, logistic regression, regularization, confusion matrix | 2025-02-21 |
| [04-neural-network-car-evaluation](problem-sets/04-neural-network-car-evaluation/neural-network-car-evaluation.ipynb) | Classifies cars into 4 acceptability classes with a Keras network: 10-unit ReLU hidden layer, 4-unit softmax output, Adam (learning rate 0.01), batch size 17, up to 50 epochs with early stopping. | Neural networks, softmax classification, early stopping, confusion matrix | 2025-03-14 |

## Methodology

All four notebooks follow the same outline: load a CSV with `pandas`, prepare the features, split 80/20 into training and test sets with `train_test_split`, fit a model on the training set, and evaluate it on the held-out test set. The details differ per problem set.

**01 · Linear regression: chickenpox**
- Data: `hungary-chickenpox.csv` (522 weekly rows: a date column plus 20 county columns). The feature `X` is column index 2 (`BARANYA`) and the target `y` is column index -3 (`VAS`).
- Split: 80/20, `random_state=1` (417 training rows, 105 test rows).
- Model: scikit-learn `LinearRegression` on the single feature, reshaped to `(-1, 1)`.
- Evaluation: mean squared error on the test set, plus scatter plots with the fitted line for the training and test sets.
- `hungary-county-edges.csv` is concatenated with the case counts into `combined_df`, but only to print a column name; it is not used by the model.

**02 · Linear regression: concrete**
- Same pipeline as problem set 1 (`random_state=1`), with `X` = column index 2 and `y` = the last column.
- As submitted the notebook reads `housing.csv` (505 rows, 404 training rows), which is not included. The bundled `concrete-data.xls` and the PDF report describe the Concrete Compressive Strength dataset; the notebook does not read that file.

**03 · Logistic regression: mushrooms**
1. Load `mushrooms.csv` (8,124 rows, 22 features plus `class`) and check for null values.
2. Cast every column to `category` and encode each with `LabelEncoder`. The target is `class`, where 0 = edible and 1 = poisonous.
3. Fit `PCA(n_components=7)` on the 22 features. The 7 components keep 85.0% of the variance. PCA is fitted on all rows before the split.
4. Split 80/20, `random_state=42` (6,499 training rows, 1,625 test rows).
5. Standardize with `StandardScaler`, fitted on the training set only and applied to both sets.
6. Fit a default `LogisticRegression`, then two regularized variants: L1 (`liblinear` solver, `C=0.5`) and L2 (`C=0.5`).
7. Evaluate with accuracy and a confusion matrix, plot predicted probability of "poisonous" against the first PCA component, and compare the three models' accuracies in a bar chart.

**04 · Neural network: car evaluation**
1. Load `car-evaluation.csv` with `pd.read_csv` defaults. The file has no header row, so the first record is read as the header (1,727 of the file's 1,728 records become data rows).
2. Encode every column with `LabelEncoder`. `X` is the first 6 columns and `y` the last (4 classes: `acc`, `good`, `unacc`, `vgood`).
3. Split 80/20, `random_state=42` (1,381 training rows, 346 test rows), then standardize with `StandardScaler` fitted on the training set.
4. Model: Keras `Sequential` with `Dense(10, activation="relu")` on 6 inputs and `Dense(4, activation="softmax")` (114 parameters).
5. Training: Adam with learning rate 0.01, `sparse_categorical_crossentropy` loss, batch size 17, up to 50 epochs, and `EarlyStopping` on the training loss (patience 10, best weights restored). No random seed is set for Keras.
6. Evaluation: `model.evaluate` on the test set, training accuracy/loss curves, and a confusion matrix built from the argmax of the predicted probabilities.

## Sample results

Output from re-running the notebooks in a scratch copy (Python 3.12.14, versions listed under [Getting started](#getting-started)) using `jupyter nbconvert --execute`. Plots are not reproduced here; they are in each notebook and PDF. A few values the notebooks only show in plots (the column names in 01, the L1/L2 accuracies and confusion matrix in 03, the epoch count and confusion matrix in 04) were printed by one diagnostic line I appended to a scratch copy; those are marked below.

<details>
<summary><b>01 · Linear regression: chickenpox</b></summary>

```text
(522,)
(522,)
name_2
(417,)
(417,)
Mean Squared Error: 377.76276265058874
```

Diagnostic line (not in the notebook): the feature column is `BARANYA` and the target column is `VAS`. The `name_2` line is printed by the notebook itself.

</details>

<details>
<summary><b>02 · Linear regression: concrete</b></summary>

This notebook fails when run as submitted (`FileNotFoundError: 'housing.csv'`), so there is no fresh output. The output saved inside the notebook from the original run was:

```text
(505,)
(505,)
(404,)
(404,)
Mean Squared Error: 143.90407874304387
```

</details>

<details>
<summary><b>03 · Logistic regression: mushrooms</b></summary>

```text
(8124, 23)
[0.33758733 0.16581098 0.12285821 0.06796611 0.05831173 0.05092539 0.04670841]
0.850168163508622
prediction 0
probability [[0.67058493 0.32941507]]
0.8344615384615385
ACC LR 0.8344615384615385
```

Diagnostic line (not in the notebook):

```text
L1 0.8344615384615385
L2 0.8344615384615385

Confusion matrix, plain logistic regression (rows = actual, columns = predicted; 0 = edible, 1 = poisonous)
[[737 106]
 [163 619]]
```

</details>

<details>
<summary><b>04 · Neural network: car evaluation</b></summary>

```text
11/11 - 0s - 16ms/step - accuracy: 0.9682 - loss: 0.0800
```

Diagnostic line (not in the notebook):

```text
epochs run 50
[[ 72   1   3   1]
 [  2  13   0   0]
 [  1   0 236   0]
 [  3   0   0  14]]      # rows = actual, columns = predicted
['acc', 'good', 'unacc', 'vgood']
```

The network is not seeded, so these numbers change from run to run. Two other runs of the same notebook gave test accuracies of 0.9653 and 0.9711. The numbers saved in the notebook and in its markdown (accuracy 0.9277, loss 0.2030) come from an earlier run.

</details>
