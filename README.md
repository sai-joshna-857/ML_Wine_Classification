# Wine Type Classification (Red vs. White) Using Logistic Regression and SVM

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## Overview
This project presents an end-to-end Machine Learning pipeline to classify wine varieties into **Red Wine (`1`)** or **White Wine (`0`)** using 11 objective physicochemical measurements. Two distinct supervised algorithms—**Logistic Regression** (a linear probabilistic baseline) and **Support Vector Machines (SVM)** (a maximum-margin classifier with an RBF kernel)—are implemented, evaluated, and compared.

---

## Dataset Description
The project utilizes the benchmark **UCI Wine Quality Dataset** (Vinho Verde):
* **Total Instances:** 6,497 samples (1,599 Red Wine, 4,898 White Wine)
* **Target Label:** `wine_type` (`1`: Red Wine, `0`: White Wine)
* **Physicochemical Features (11 Attributes):**
  1. `fixed acidity`
  2. `volatile acidity`
  3. `citric acid`
  4. `residual sugar`
  5. `chlorides`
  6. `free sulfur dioxide`
  7. `total sulfur dioxide`
  8. `density`
  9. `pH`
  10. `sulphates`
  11. `alcohol`

*(Note: The subjective `quality` score given by sensory human panels is excluded to focus strictly on physical and chemical classification).*

---

## Preprocessing Pipeline
1. **Data Integration & Cleaning:** Combined separate red and white datasets, verified zero missing values, and removed duplicate entries.
2. **Stratified Splitting:** Partitioned the data into an 80/20 train-test split with stratification (`stratify=y`) to preserve the class balance ratio.
3. **Feature Standardization:** Applied `StandardScaler` (Z-score normalization) fitted strictly on the training set to prevent data leakage and ensure equal feature weighting for distance/gradient calculations.

---

## Repository Structure

```text
├── 01_Dataset_Exploration.ipynb               # Data loading and initial inspection
├── 02_EDA.ipynb                               # Visualizations, distributions, and correlation heatmap
├── 03_Final_Preprocessing_Complete_Steps.ipynb # Cleaning, scaling, and CSV split generation
├── 04_Logistic_Regression.ipynb               # Logistic Regression model training and evaluation
├── 05_SVM_Classification.ipynb                # SVM (RBF kernel) model training and evaluation
├── 06_Model_Comparison_and_Evaluation.ipynb   # Comparative ROC curves and prediction export
├── DATASET_PROJECT.csv                        # Combined raw dataset
├── X_train_processed.csv                      # Standardized training features
├── X_test_processed.csv                       # Standardized testing features
├── y_train.csv                                # Training ground truth labels
├── y_test.csv                                 # Testing ground truth labels
├── Prediction.csv                             # Model predictions & class probabilities
└── README.md                                  # Project documentation
