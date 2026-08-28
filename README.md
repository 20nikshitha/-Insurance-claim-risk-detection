# Automobile Insurance Fraud Detection using Machine Learning

## 📌 Project Overview

This project develops a machine learning-based system for detecting potentially fraudulent automobile insurance claims.

The main objective is to identify fraudulent claims from historical insurance data and help insurance companies prioritize suspicious claims for further investigation.

Since fraudulent claims represent only a small portion of the dataset, the project focuses on metrics such as Recall, Precision, F1 Score, and ROC-AUC rather than relying only on accuracy.

---
## ⭐ Key Results

The Gradient Boosting model was selected as the final model.

- ROC-AUC: **84.45%**
- Recall: **72.97%**
- F1 Score: **28.18%**
- Precision: **17.46%**
- Classification Threshold: **0.10**

The model is designed to flag potentially fraudulent claims for further investigation rather than automatically rejecting claims.

## 📊 Dataset

The dataset contains:

- 15,420 insurance claim records
- 33 original features
- Target variable: `FraudFound`

### Target Distribution

- Not Fraud: 14,497 (94.01%)
- Fraud: 923 (5.99%)

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

---

## 🔎 Exploratory Data Analysis

The dataset was analyzed to understand:

- Fraud vs. non-fraud distribution
- Policy types and their fraud rates
- Claim patterns
- Vehicle and policy characteristics
- Categorical and numerical features
- Potential relationships between features and fraudulent claims

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Checked for duplicate records
- Identified missing values
- Handled invalid values such as `0` in claim-related categorical fields
- Removed `PolicyNumber` as it acts as an identifier rather than a meaningful predictive feature
- Created a binary target variable `FraudFlag`
- Applied preprocessing separately to numerical and categorical features
- Used a preprocessing pipeline for machine learning

---

## 🤖 Machine Learning Models

Three classification models were developed and compared:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Because of the strong class imbalance, classification thresholds were tuned to improve fraud detection.

---

## 📈 Model Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 64.33% | 13.25% | 89.19% | 23.08% | 80.58% |
| Random Forest | 0.10 | 76.85% | 16.56% | 70.81% | 26.84% | 83.55% |
| Gradient Boosting | 0.10 | 77.69% | 17.46% | 72.97% | 28.18% | 84.45% |

---

## 🏆 Final Model

Gradient Boosting was selected as the final model.

### Final Performance

- Accuracy: **77.69%**
- Precision: **17.46%**
- Recall: **72.97%**
- F1 Score: **28.18%**
- ROC-AUC: **84.45%**
- Classification Threshold: **0.10**

The threshold was reduced from the default 0.50 to 0.10 to improve the model's ability to identify fraudulent claims.

The final model is intended as a fraud-screening system that can flag potentially suspicious claims for further investigation.

---

## 🔍 Feature Importance

The Gradient Boosting model identified several influential features, including:

- AddressChange-Claim
- BasePolicy
- Fault
- Age
- PolicyType
- Year
- Vehicle-related characteristics

Feature importance indicates which variables contributed most to the model's predictions. It does not necessarily imply that these variables directly cause fraud.

---

## 🧪 New Claim Prediction

The trained model was saved and tested on a new insurance claim.

Example prediction:

```text
Fraud Probability: 0.0142
Threshold: 0.10
Prediction: NOT FRAUD
