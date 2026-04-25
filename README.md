# 🤖 Machine Learning Portfolio — Smit Satani

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-enabled-green?logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/License-No%20License%20Specified-lightgrey" />
</p>

<p align="center">
  <em>A hands-on collection of end-to-end Machine Learning projects spanning classification, regression, clustering, NLP, and hyperparameter optimization — built to learn, experiment, and showcase real-world ML workflows.</em>
</p>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [ML Tasks Covered](#-ml-tasks-covered)
3. [Repository Structure](#-repository-structure)
4. [Setup Instructions](#️-setup-instructions)
5. [Projects & How to Run](#-projects--how-to-run)
   - [Credit Card Fraud Detection](#1-credit-card-fraud-detection)
   - [Default Credit Card Payment (SVC Pipeline)](#2-default-credit-card-payment-svc-pipeline)
   - [Titanic Survival Prediction](#3-titanic-survival-prediction)
   - [CO₂ Emission Prediction with Pipeline](#4-co-emission-prediction-with-pipeline)
   - [Disease Detection with Boosting Algorithms](#5-disease-detection-with-boosting-algorithms)
   - [Clustering (KMeans, Food Delivery, Fashion-MNIST)](#6-clustering)
   - [Decision Tree (Basic + Hyperparameter Tuning)](#7-decision-tree)
   - [Simple Linear Regression](#8-simple-linear-regression)
   - [NLP — Resume Screening App](#9-nlp--resume-screening-app)
   - [Optuna — Hyperparameter Optimization Basics](#10-optuna--hyperparameter-optimization-basics)
6. [Datasets Used](#-datasets-used)
7. [Results Summary](#-results-summary)
8. [Reproducibility Notes](#-reproducibility-notes)
9. [Roadmap / Next Steps](#️-roadmap--next-steps)
10. [Contributing](#-contributing)
11. [License](#-license)

---

## 🌟 Project Overview

This repository is a growing **Machine Learning portfolio** that covers a wide spectrum of real-world problems — from detecting financial fraud to screening resumes with NLP. Every project follows the same disciplined workflow:

```
Data Ingestion → EDA → Preprocessing → Feature Engineering
     → Model Training → Evaluation → (Hyperparameter Tuning)
```

The goals are to:
- Practise and demonstrate core ML concepts on real datasets.
- Compare multiple algorithms side-by-side with quantitative metrics.
- Build clean, reproducible Jupyter notebooks that serve as study material.
- Progressively incorporate best-practice tooling (Pipelines, Optuna, SMOTE, VIF, etc.).

---

## 🧠 ML Tasks Covered

| Task | Techniques Used | Notebooks |
|---|---|---|
| **Binary Classification** | Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVC, XGBoost, LightGBM, CatBoost, AdaBoost, GradientBoosting | Fraud Detection, Titanic, Disease Detection |
| **Regression** | Linear Regression, Ridge, Lasso, Random Forest Regressor, Gradient Boosting Regressor | CO₂ Emission, Advertising Sales |
| **Clustering** (Unsupervised) | K-Means (Elbow + Silhouette), PCA-based visualisation | KMeans basics, Food Delivery segmentation, Fashion-MNIST |
| **Dimensionality Reduction** | PCA | Fashion-MNIST |
| **NLP / Text Classification** | TF-IDF, CountVectorizer, K-Nearest Neighbours, SVM | Resume Screening |
| **Hyperparameter Optimisation** | Optuna (TPE, Random Sampler, Grid Sampler, CMA-ES), GridSearchCV | Optuna basics, Decision Tree tuning |
| **Imbalanced Learning** | SMOTE, class_weight balancing | Credit Card Fraud, Default Payment |
| **ML Pipelines** | `sklearn.pipeline.Pipeline`, `ColumnTransformer` | CO₂ Emission, Default Credit Card |

---

## 📁 Repository Structure

```
ML/
│
├── Boosting_Disease_Detection.ipynb      # Ensemble boosting models for heart-disease detection
├── Titanic_Survival_Prediction.ipynb     # Classic Titanic classification with Logistic Reg + RF
├── co2-emission-with-pipeline.ipynb      # Regression with full sklearn Pipeline
├── Optuna_basics.ipynb                   # Hyperparameter optimization with Optuna
│
├── Clustering/
│   ├── KMeans_done.ipynb                 # KMeans fundamentals (synthetic blobs)
│   ├── Food_delivery.ipynb               # Customer segmentation for a food-delivery app
│   ├── fashion-mnist.ipynb               # KMeans + KNN on Fashion-MNIST (PCA-reduced)
│   └── food_delivery (1).csv             # Food-delivery dataset
│
├── Credit_Card_Fraud_Detection/
│   ├── Credit_Card_Fraud_Detection.ipynb # Multi-model fraud detection + class imbalance
│   ├── default-credit-card-svc_pipeline.ipynb  # SVC Pipeline + VIF + SMOTE
│   ├── XGB_model.pkl                     # Saved XGBoost model
│   ├── dt_model.pkl                      # Saved Decision Tree model
│   ├── lr_model.pkl                      # Saved Logistic Regression model
│   └── rfc_model.pkl                     # Saved Random Forest model
│
├── DecisionTree/
│   ├── decision-tree-basic.ipynb         # Decision Tree on Car Evaluation dataset
│   └── decision-tree-hyperparameter.ipynb # GridSearchCV tuning on Titanic
│
├── Linear_regression/
│   └── simple/
│       ├── simple-linear-regression-using-sklearn (1).ipynb  # TV ad spend → Sales
│       └── advertising.csv               # Advertising dataset
│
└── NLP/
    └── nlp-resume-screening-app.ipynb    # Resume category classification with TF-IDF
```

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python ≥ 3.9**
- `pip` or `conda`

### 1 — Clone the repository

```bash
git clone https://github.com/satanismit/ML.git
cd ML
```

### 2 — Create a virtual environment

**Using `venv`:**
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

**Using `conda`:**
```bash
conda create -n ml-portfolio python=3.10 -y
conda activate ml-portfolio
```

### 3 — Install dependencies

All notebooks share a common dependency set. Install everything at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn \
            xgboost lightgbm catboost optuna imbalanced-learn \
            statsmodels openpyxl notebook
```

> **Note:** `catboost` can be slow to install on some platforms. If you don't need it, omit it — it is only used in `Boosting_Disease_Detection.ipynb`.

### 4 — Launch Jupyter

```bash
jupyter notebook
```

Then navigate to the notebook you want to run.

---

## 🚀 Projects & How to Run

### 1. Credit Card Fraud Detection

**Notebook:** `Credit_Card_Fraud_Detection/Credit_Card_Fraud_Detection.ipynb`

**Problem:** Detect fraudulent transactions in a highly imbalanced dataset (fraud ≈ 0.17 % of all transactions).

**Workflow:**
- Load the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`).
- Exploratory Data Analysis — class distribution pie chart, feature correlations.
- Data scaling with `StandardScaler` on the `Amount` and `Time` features.
- Train and compare **6 models**: Logistic Regression, Decision Tree, Naive Bayes, Random Forest, Gradient Boosting, XGBoost.
- Evaluate with accuracy, precision, recall, F1-score, and classification report.
- Saved models: `lr_model.pkl`, `dt_model.pkl`, `rfc_model.pkl`, `XGB_model.pkl`.

**How to run:**
1. Download `creditcard.csv` from Kaggle and place it in `Credit_Card_Fraud_Detection/`.
2. Open and run all cells in `Credit_Card_Fraud_Detection.ipynb`.

---

### 2. Default Credit Card Payment (SVC Pipeline)

**Notebook:** `Credit_Card_Fraud_Detection/default-credit-card-svc_pipeline.ipynb`

**Problem:** Predict whether a credit card client will default on their next payment.

**Workflow:**
- Dataset: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (`.xls`).
- Feature engineering: compute `BILL_AVG` (average billing amount over 6 months) to reduce multicollinearity.
- **VIF (Variance Inflation Factor)** analysis to detect and remove multicollinear features.
- **SMOTE** oversampling to handle class imbalance.
- Full sklearn `Pipeline` with `PowerTransformer` + `StandardScaler` + model.
- Comparison of **Decision Tree**, **Logistic Regression**, **Random Forest**, **SVC**, and **XGBoost**.

**How to run:**
1. Download the `.xls` file from UCI / Kaggle and update the file path in the notebook.
2. Open and run all cells.

---

### 3. Titanic Survival Prediction

**Notebook:** `Titanic_Survival_Prediction.ipynb`

**Problem:** Predict passenger survival on the Titanic — the quintessential binary-classification benchmark.

**Workflow:**
- Load `train.csv` from [Kaggle Titanic](https://www.kaggle.com/c/titanic/data).
- Missing value imputation: mean for `Age`, mode for `Embarked`, drop `Cabin`.
- Label-encode categorical columns (`Sex`, `Embarked`).
- Train **Logistic Regression** and **Random Forest Classifier**.
- Evaluate on held-out test split.

**How to run:**
1. Download `train.csv` from Kaggle and update the path in the notebook.
2. Open and run all cells.

---

### 4. CO₂ Emission Prediction with Pipeline

**Notebook:** `co2-emission-with-pipeline.ipynb`

**Problem:** Predict CO₂ emissions (g/km) from vehicle specifications (regression).

**Workflow:**
- Dataset: [CO2 Emissions Canada — Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles).
- EDA: manufacturer distribution, feature correlations.
- Label-encode `Make`, `Model`, `Vehicle Class`, `Fuel Type`, `Transmission`.
- Build a `ColumnTransformer` + `Pipeline` approach to compare:
  - Linear Regression, Ridge, Lasso
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Metrics: MAE, MSE, R² score.

**How to run:**
1. Download the dataset from Kaggle and update the path.
2. Open and run all cells.

---

### 5. Disease Detection with Boosting Algorithms

**Notebook:** `Boosting_Disease_Detection.ipynb`

**Problem:** Predict heart disease from patient health indicators using ensemble boosting models.

**Workflow:**
- Load `train.csv` (large dataset, ~630 k training rows).
- Custom `preprocess()` function: drop duplicates, impute, scale, encode.
- Compare **5 ensemble methods**: Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost.
- All wrapped in a `sklearn Pipeline` with a `ColumnTransformer` for numeric + categorical features.
- Metrics: Accuracy, F1-Score, Precision, Recall.

**How to run:**
```bash
pip install xgboost lightgbm catboost
```
1. Place `train.csv` in the repository root (or update the path).
2. Open and run all cells.

---

### 6. Clustering

#### 6a. KMeans Fundamentals
**Notebook:** `Clustering/KMeans_done.ipynb`

- Synthetic blob dataset (`make_blobs`, 1 000 samples, 3 clusters).
- **Elbow Method** (WCSS) to choose optimal K.
- Silhouette Score validation.
- Visualise clusters with scatter plots.

#### 6b. Food Delivery Customer Segmentation
**Notebook:** `Clustering/Food_delivery.ipynb`

- Dataset: `food_delivery (1).csv` (committed to the repo).
- Segment food-delivery app users by behaviour/demographics.
- EDA, null-value handling, feature scaling.
- KMeans clustering, cluster profiling, and business insights.

#### 6c. Fashion-MNIST Clustering + KNN
**Notebook:** `Clustering/fashion-mnist.ipynb`

- Dataset: [Fashion-MNIST — Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
- Normalise pixel values to [0, 1].
- **PCA** dimensionality reduction before clustering.
- K-Means with Silhouette, ARI, NMI evaluation.
- **KNN** classification on PCA-reduced features (K=1–20 accuracy sweep).

**How to run (6a):**
```bash
# No external data needed — synthetic data generated inside the notebook
```

**How to run (6b):**
- `food_delivery (1).csv` is already in `Clustering/`. Just open and run.

**How to run (6c):**
1. Download `fashion-mnist_train.csv` from Kaggle and update the path.
2. Open and run all cells.

---

### 7. Decision Tree

#### 7a. Decision Tree Basics
**Notebook:** `DecisionTree/decision-tree-basic.ipynb`

- Dataset: [Car Evaluation — UCI](https://archive.ics.uci.edu/dataset/19/car+evaluation) (`car.data`).
- All-categorical features → `LabelEncoder` for all columns.
- Decision Tree trained and visualised.
- Feature importance analysed via entropy/information gain.

#### 7b. Decision Tree with Hyperparameter Tuning
**Notebook:** `DecisionTree/decision-tree-hyperparameter.ipynb`

- Dataset: Titanic (`Titanic-Dataset.csv`).
- Missing-value handling, label-encoding.
- `GridSearchCV` over `criterion`, `max_depth`, `max_features`, `min_samples_leaf`, `min_samples_split`.
- Best parameters selected; final model evaluated.

**How to run:**
1. Download the respective dataset and update the path.
2. Open and run all cells.

---

### 8. Simple Linear Regression

**Notebook:** `Linear_regression/simple/simple-linear-regression-using-sklearn (1).ipynb`

**Problem:** Predict product sales from TV advertising spend.

**Workflow:**
- Dataset: `advertising.csv` (committed in `Linear_regression/simple/`).
- Correlation heatmap to select the best predictor (`TV` → `Sales`).
- Fit `LinearRegression` model.
- Evaluate with MAE, MAPE, MSE.
- Visualise: Actual vs Predicted plots, regression line.

**How to run:**
```bash
# Dataset is already committed — just open and run all cells.
```

---

### 9. NLP — Resume Screening App

**Notebook:** `NLP/nlp-resume-screening-app.ipynb`

**Problem:** Automatically classify resumes into 25 job categories (Data Science, HR, Blockchain, etc.).

**Workflow:**
- Dataset: [Updated Resume Dataset — Kaggle](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset) (962 resumes, 25 categories).
- Text cleaning: remove URLs, special characters, stopwords.
- **TF-IDF** and **CountVectorizer** vectorisation.
- **SMOTE** oversampling for class balance.
- Classifiers: **K-Nearest Neighbours (KNN)** and **SVM**.
- Achieve ≈ 99.5 % accuracy on the test set.

**How to run:**
1. Download `UpdatedResumeDataSet.csv` from Kaggle and update the path.
2. Open and run all cells.

---

### 10. Optuna — Hyperparameter Optimization Basics

**Notebook:** `Optuna_basics.ipynb`

**Problem:** Learn and demonstrate Optuna's hyperparameter search framework on the Diabetes/Pima Indians dataset.

**Workflow:**
- Dataset: [Pima Indians Diabetes — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- Replace biologically impossible zero values with `NaN`, then impute.
- Define an Optuna `objective` function for `RandomForestClassifier`.
- Run 50-trial studies with **4 different samplers**:
  - Default (TPE), `RandomSampler`, `GridSampler`, `CmaEsSampler`.
- Compare best accuracy and hyperparameters across samplers.
- Bonus: Optuna study for **XGBoost** on a separate dataset.

**How to run:**
```bash
pip install optuna
```
1. Download the dataset and update the path.
2. Open and run all cells.

---

## 📊 Datasets Used

| Dataset | Source | License | Location |
|---|---|---|---|
| Credit Card Fraud | [Kaggle / ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Open Database License (ODbL) | Download from Kaggle |
| Default of Credit Card Clients | [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | CC BY 4.0 | Download from UCI / Kaggle |
| Titanic | [Kaggle](https://www.kaggle.com/c/titanic/data) | Kaggle Rules | Download from Kaggle |
| CO₂ Emissions Canada | [Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) | Open Government License | Download from Kaggle |
| Heart Disease (Boosting) | Kaggle (`train.csv`) | See Kaggle competition rules | Download from Kaggle |
| Food Delivery | Custom / Online | Unknown | Committed: `Clustering/food_delivery (1).csv` |
| Fashion-MNIST | [Kaggle / Zalando](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | MIT | Download from Kaggle |
| Car Evaluation | [UCI ML Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation) | CC BY 4.0 | Download from UCI |
| Advertising | Custom (`advertising.csv`) | — | Committed: `Linear_regression/simple/advertising.csv` |
| Pima Indians Diabetes | [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | CC0: Public Domain | Download from Kaggle |
| Updated Resume Dataset | [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset) | Unknown | Download from Kaggle |

> **Datasets not committed** (size constraints): `creditcard.csv`, CO₂ Canada, Titanic train/test, Heart Disease train, Fashion-MNIST, Car Evaluation, Pima Diabetes, Resume Dataset. Download from the links above and update the file paths inside each notebook.

---

## 📈 Results Summary

### Classification

| Project | Best Model | Accuracy | F1-Score | Notes |
|---|---|---|---|---|
| **Credit Card Fraud** | Random Forest | **99.96 %** | — | Highly imbalanced (0.17 % fraud) |
| | XGBoost | 99.95 % | 0.83 (fraud class) | Precision 0.92, Recall 0.75 |
| | Logistic Regression | 99.92 % | — | |
| **Default Credit Card** | Decision Tree | 74.3 % | 0.743 | Balanced via SMOTE |
| | Logistic Regression | 72.1 % | 0.721 | |
| **Titanic** | Random Forest | **94.4 %** | — | Train 98.3 % |
| | Logistic Regression | 80.6 % | — | |
| **Disease Detection** | XGBoost | **88.7 %** | 0.886 | 630 k training rows |
| | LightGBM | 88.7 % | 0.885 | |
| | Gradient Boosting | 88.6 % | 0.885 | |
| | AdaBoost | 88.5 % | 0.883 | |
| | Random Forest | 88.0 % | 0.879 | |
| **Decision Tree (Titanic)** | DT + GridSearchCV | 82.8 % | — | Best: gini, max_depth=3 |

### Regression

| Project | Best Model | R² Score | MAE | MSE |
|---|---|---|---|---|
| **CO₂ Emission** | Gradient Boosting | **0.9977** | 1.80 | 8.02 |
| | Random Forest | 0.9972 | 1.82 | 9.71 |
| | Ridge/Lasso | ~0.987 | ~4.67 | ~45.9 |
| | Linear Regression | 0.907 | 11.18 | 295.3 |
| **Advertising Sales** | Linear Regression | — | — | Visualised actual vs predicted |

### Clustering (Fashion-MNIST)

| Metric | Score |
|---|---|
| Silhouette Score (PCA + KMeans) | 0.197 |
| Adjusted Rand Index (ARI) | 0.397 |
| Normalised Mutual Info (NMI) | 0.551 |
| KNN Best Accuracy (K=8) | **85.3 %** |

### NLP (Resume Screening)

| Model | Accuracy | F1-Score |
|---|---|---|
| KNN + TF-IDF | **99.5 %** | 0.995 |
| KNN + CountVectorizer | **99.5 %** | 0.995 |
| SVM + TF-IDF | **98.4 %** | 0.984 |

### Optuna (Diabetes Dataset)

| Sampler | Best CV Accuracy | Test Accuracy |
|---|---|---|
| TPE (default) | 78.2 % | 74 % |
| RandomSampler | 78.0 % | 75 % |
| GridSampler | 77.5 % | 74 % |
| CmaEsSampler | 78.9 % (SVM) | — |

---

## 🔁 Reproducibility Notes

| Parameter | Details |
|---|---|
| **Python version** | 3.9 – 3.13 (developed on 3.13 / Anaconda) |
| **Random seeds** | `random_state=42` used in all `train_test_split` and model constructors |
| **Hardware** | CPU-only; no GPU required for any notebook |
| **OS** | Developed on Windows 11 (`win_amd64`); tested-compatible with Linux/macOS |
| **Key library versions** | `xgboost==3.2.0`, `lightgbm==4.6.0`, `catboost==1.2.8`, `optuna` (latest) |
| **Data splits** | Typically 70 / 30 or 75 / 25 train / test splits with fixed seed |
| **SMOTE** | Applied only to training data after the split to prevent data leakage |

To pin the exact environment:
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

---

## 🗺️ Roadmap / Next Steps

- [ ] **Deep Learning** — Add CNNs (Keras/PyTorch) for Fashion-MNIST image classification.
- [ ] **Time Series** — Stock price or weather forecasting with LSTM / Prophet.
- [ ] **Model Deployment** — Wrap the best models (Fraud, Resume) in a Flask / FastAPI service.
- [ ] **MLflow / Weights & Biases** — Experiment tracking and model registry.
- [ ] **Feature Importance** — SHAP values for interpretability on boosting models.
- [ ] **Automated EDA** — Integrate `ydata-profiling` for one-click dataset reports.
- [ ] **CI/CD** — GitHub Actions to run notebooks on push (Papermill or nbconvert).
- [ ] **requirements.txt** — Add pinned dependency file for 100 % reproducible installs.

---

## 🤝 Contributing

Contributions, improvements, and new notebooks are warmly welcomed!

1. **Fork** the repository and create your feature branch:
   ```bash
   git checkout -b feature/your-topic
   ```
2. Add your notebook following the existing naming convention.
3. Make sure your notebook **runs top-to-bottom without errors** (`Kernel → Restart & Run All`).
4. Include a brief **markdown summary cell** at the top of your notebook describing the goal and dataset.
5. Open a **Pull Request** with a clear description of what you added.

---

## 📄 License

**No license specified.**

All notebooks are provided for educational and portfolio purposes. The datasets used belong to their respective owners / platforms (Kaggle, UCI ML Repository). Please refer to each dataset's individual license before redistribution.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/satanismit">Smit Satani</a>
</p>
