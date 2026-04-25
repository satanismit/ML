# Machine Learning Projects

A personal collection of hands-on machine learning projects covering supervised learning, unsupervised learning, NLP, and hyperparameter tuning — built using Python, scikit-learn, and related libraries.

---

## Repository Structure

```
ML/
│
├── Boosting_Disease_Detection.ipynb       # Heart disease detection with boosting algorithms
├── Titanic_Survival_Prediction.ipynb      # Titanic survival prediction (classification)
├── co2-emission-with-pipeline.ipynb       # CO₂ emission prediction with sklearn Pipeline
├── Optuna_basics.ipynb                    # Hyperparameter tuning with Optuna
│
├── Clustering/
│   ├── Food_delivery.ipynb                # User segmentation with KMeans + PCA
│   ├── KMeans_done.ipynb                  # KMeans from scratch on synthetic data
│   ├── fashion-mnist.ipynb                # Fashion MNIST with KNN, KMeans, TSNE, DBSCAN
│   └── food_delivery (1).csv             # Dataset for food delivery clustering
│
├── Credit_Card_Fraud_Detection/
│   ├── Credit_Card_Fraud_Detection.ipynb  # Fraud detection with multiple classifiers
│   ├── default-credit-card-svc_pipeline.ipynb  # Default payment prediction with SVC pipeline
│   ├── XGB_model.pkl                      # Saved XGBoost model
│   ├── dt_model.pkl                       # Saved Decision Tree model
│   ├── lr_model.pkl                       # Saved Logistic Regression model
│   └── rfc_model.pkl                      # Saved Random Forest model
│
├── DecisionTree/
│   ├── decision-tree-basic.ipynb          # Decision Tree on car evaluation dataset
│   └── decision-tree-hyperparameter.ipynb # Decision Tree with GridSearchCV on Titanic
│
├── Linear_regression/
│   └── simple/
│       ├── simple-linear-regression-using-sklearn (1).ipynb  # Simple linear regression (TV vs Sales)
│       └── advertising.csv                # Advertising dataset
│
└── NLP/
    └── nlp-resume-screening-app.ipynb     # Resume category classification with NLP + BERT
```

---

## Projects Overview

### 1. Boosting — Disease Detection
**File:** `Boosting_Disease_Detection.ipynb`

Predicts heart disease using five boosting/ensemble models inside a scikit-learn `Pipeline`. The pipeline handles preprocessing (imputation, encoding, scaling) and then trains each model.

**Models compared:**
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

**Key techniques:** `Pipeline`, `GridSearchCV` for hyperparameter tuning, metrics: accuracy, precision, recall, F1-score.

---

### 2. Titanic Survival Prediction
**File:** `Titanic_Survival_Prediction.ipynb`

Classic binary classification problem — predicts whether a passenger survived the Titanic disaster.

**Steps covered:**
- Data cleaning and null value handling
- Feature engineering
- Training with Logistic Regression and Random Forest
- Evaluation of train/test accuracy to compare model performance

---

### 3. CO₂ Emission Prediction with Pipeline
**File:** `co2-emission-with-pipeline.ipynb`

Predicts CO₂ emissions from vehicles (regression task) using the Canada CO₂ Emissions dataset from Kaggle.

**Models compared:**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNeighbors Regressor

**Key techniques:** `Pipeline` for preprocessing + modelling, model comparison, saving the final pipeline.

---

### 4. Optuna — Hyperparameter Tuning
**File:** `Optuna_basics.ipynb`

Demonstrates how to use Optuna (an automatic hyperparameter optimization framework) to tune a Random Forest classifier on the Pima Indians Diabetes dataset.

**Covers:**
- Defining an `objective` function
- Using Optuna `study` with different samplers (default TPE, Random)
- Running 50 trials to find best hyperparameters
- Optimizing multiple ML models
- Optuna built-in visualizations (optimization history, parameter importance)

---

### 5. Clustering — Food Delivery User Segmentation
**File:** `Clustering/Food_delivery.ipynb`  
**Dataset:** `Clustering/food_delivery (1).csv`

Segments food delivery app users into groups based on their behavior (total orders, average spending, app usage time, delivery ratings, cuisine preferences).

**Steps:**
- EDA and summary statistics
- Feature scaling with StandardScaler
- Dimensionality reduction with PCA (2D visualization)
- KMeans clustering (3 clusters)
- Cluster visualization and interpretation

---

### 6. KMeans — Basics
**File:** `Clustering/KMeans_done.ipynb`

Applies KMeans on a synthetic dataset (`make_blobs`) to understand how the algorithm works fundamentally.

**Covers:** Creating blobs, training KMeans, evaluating with silhouette score, visualizing clusters.

---

### 7. Fashion MNIST — Clustering & Classification
**File:** `Clustering/fashion-mnist.ipynb`

Applies multiple unsupervised and supervised algorithms to the Fashion MNIST image dataset.

**Techniques used:**
- KNN (without scaling vs. with StandardScaler + PCA)
- KMeans clustering
- t-SNE for 2D visualization of high-dimensional image data
- DBSCAN
- Agglomerative Clustering

---

### 8. Credit Card Fraud Detection
**File:** `Credit_Card_Fraud_Detection/Credit_Card_Fraud_Detection.ipynb`

Detects fraudulent credit card transactions on a highly imbalanced dataset (0.17% fraud rate).

**Key observations:**
- Logistic Regression gave 99.9% accuracy (misleading due to class imbalance)
- Used confusion matrix as the real evaluation metric
- Decision Tree reduced false negatives from 44 → 25 compared to Logistic Regression
- Also tested Naive Bayes (performed poorly)
- Saved trained models: `lr_model.pkl`, `dt_model.pkl`, `rfc_model.pkl`, `XGB_model.pkl`

---

### 9. Default Credit Card Payment — SVC Pipeline
**File:** `Credit_Card_Fraud_Detection/default-credit-card-svc_pipeline.ipynb`

Predicts whether a customer will default on their credit card payment next month.

**Key techniques:**
- VIF (Variance Inflation Factor) to detect and handle multicollinearity
- Feature engineering: computed average billing across 6 months (`BILL_AVG`)
- Handled class imbalance
- Built a `Pipeline` with `StandardScaler` + `PowerTransformer` + SVC
- Compared models without tuning vs. with hyperparameter tuning

---

### 10. Decision Tree — Basic
**File:** `DecisionTree/decision-tree-basic.ipynb`

Applies a Decision Tree classifier to the Car Evaluation dataset from Kaggle.

**Steps:** Label encoding all categorical columns → training Decision Tree → plotting the tree → evaluating accuracy.

---

### 11. Decision Tree — Hyperparameter Tuning
**File:** `DecisionTree/decision-tree-hyperparameter.ipynb`

Applies `GridSearchCV` to find the best hyperparameters for a Decision Tree on the Titanic dataset.

**Covers:** Null handling, feature selection, encoding, and tuning `max_depth`, `min_samples_split`, `criterion`.

---

### 12. Simple Linear Regression
**File:** `Linear_regression/simple/simple-linear-regression-using-sklearn (1).ipynb`  
**Dataset:** `Linear_regression/simple/advertising.csv`

Predicts product sales from TV advertising spend.

**Steps:**
- Correlation heatmap to select the best predictor (`TV`)
- Scatter plot (TV vs Sales)
- Train/test split (70/30)
- LinearRegression from sklearn
- Evaluation with MAE, MAPE, MSE

---

### 13. NLP — Resume Screening
**File:** `NLP/nlp-resume-screening-app.ipynb`

Classifies resumes into job categories (e.g., Data Science, HR, Finance) using NLP techniques.

**Pipeline:**
- Text cleaning (removing special characters, stopwords)
- TF-IDF vectorization
- SMOTE to handle class imbalance
- Label encoding for target categories
- Models compared: Logistic Regression, Random Forest
- State-of-the-art: BERT embeddings + Logistic Regression

---

## Setup & Running

All notebooks are standard Jupyter notebooks. To run them locally:

```bash
# Install common dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna imbalanced-learn transformers jupyterlab

# Launch Jupyter
jupyter lab
```

> **Note:** Some notebooks were originally run on Kaggle or Google Colab, so dataset paths like `/kaggle/input/...` or `/content/...` may need to be updated to your local paths.

---

## Key Techniques Used

| Technique | Where Used |
|---|---|
| sklearn Pipelines | CO₂ Emission, Boosting, Credit Card Default |
| GridSearchCV | Boosting, Decision Tree Hyperparameter |
| Optuna (Bayesian tuning) | Optuna Basics |
| KMeans + PCA | Food Delivery, Fashion MNIST, KMeans Basics |
| t-SNE | Fashion MNIST |
| DBSCAN / Agglomerative | Fashion MNIST |
| SMOTE (imbalance handling) | NLP Resume Screening, Credit Card Default |
| VIF (multicollinearity) | Credit Card Default |
| TF-IDF + BERT | NLP Resume Screening |
| Model persistence (pickle) | Credit Card Fraud Detection |
