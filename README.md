# 🔍 Insurance Claim Fraud Detection & Risk Assessment

An end-to-end **Machine Learning + Explainable AI + Generative AI** system for detecting potentially fraudulent insurance claims and converting model predictions into actionable, investigator-friendly risk insights.

The project combines a **Gradient Boosting classifier**, **SHAP explainability**, and an **LLM-powered explanation layer** inside an interactive **Streamlit application**.

---

## 🚀 Project Overview

Insurance fraud can result in significant financial losses and inefficient use of investigation resources. Traditional rule-based systems can identify obvious suspicious patterns, but they may struggle with complex relationships across multiple claim attributes.

This project addresses the problem by building a machine-learning pipeline that:

- Predicts the probability of insurance claim fraud
- Classifies claims into **Low, Medium, and High Risk**
- Identifies the features influencing each prediction
- Uses **SHAP** to provide transparent model explanations
- Uses an **LLM** to convert technical model outputs into simple investigator-friendly explanations
- Allows users to upload claim data through a Streamlit interface
- Provides downloadable scored claims for further investigation

The goal is not to automatically declare a claim fraudulent, but to provide a **risk-prioritization and decision-support system** for human investigators.

---

## 🎯 Key Objectives

### 1. Fraud Risk Prediction
Train a supervised machine-learning model to estimate the likelihood that an insurance claim is fraudulent.

### 2. Risk Prioritization
Convert model probabilities into three practical risk categories:

| Fraud Probability | Risk Category |
|---|---|
| `< 10%` | 🟢 Low Risk |
| `10% – <30%` | 🟡 Medium Risk |
| `≥ 30%` | 🔴 High Risk |

### 3. Explainable AI
Use SHAP (SHapley Additive exPlanations) to identify which features contributed most strongly to an individual prediction.

### 4. Generative AI
Use an LLM as an interpretation layer that transforms SHAP factors and model output into a concise, professional explanation.

### 5. Interactive Deployment
Provide an easy-to-use Streamlit application for claim scoring and investigation support.

---

# 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │   Insurance Claims  │
                │       Dataset       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Preprocessing  │
                │                     │
                │ • Missing Values    │
                │ • Numerical Features│
                │ • Categorical Data  │
                │ • One-Hot Encoding  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Gradient Boosting   │
                │     Classifier      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Fraud Probability   │
                └──────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐       ┌──────────────────┐
     │ Risk Category   │       │ SHAP Explainability│
     │                 │       │                  │
     │ Low / Medium /  │       │ Feature Impact   │
     │ High            │       │                  │
     └────────┬────────┘       └────────┬─────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │   LLM Explanation   │
                │                     │
                │ Investigator-       │
                │ Friendly Summary    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Streamlit App     │
                │                     │
                │ • Risk Dashboard    │
                │ • Claim Analysis    │
                │ • SHAP Factors      │
                │ • AI Explanation    │
                │ • CSV Download      │
                └─────────────────────┘
