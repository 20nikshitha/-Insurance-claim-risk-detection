import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

THRESHOLD = 0.10

# The code checks several possible locations so the app is
# less likely to fail because of the folder name.
BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "insurance_fraud_gradient_boosting.pkl",
    BASE_DIR / "MODELS" / "insurance_fraud_gradient_boosting.pkl",
    BASE_DIR / "models" / "insurance_fraud_gradient_boosting.pkl",
]


# ============================================================
# HEADER
# ============================================================

st.title("🔍 Insurance Claim Fraud Detection System")

st.markdown(
    """
This application uses a **Gradient Boosting machine-learning model**
to estimate insurance claim fraud risk.

### Features

- Fraud probability
- Fraud prediction using a 0.10 threshold
- Low / Medium / High risk classification
- Individual claim analysis
- SHAP-based explanations
- Investigator summary
- AI-generated claim explanation
- Scored claim download
"""
)

st.divider()


# ============================================================
# FIND MODEL
# ============================================================

def find_model_path():

    for path in MODEL_CANDIDATES:

        if path.exists():
            return path

    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


model_path = find_model_path()

if model_path is None:

    st.error("❌ Trained model file was not found.")

    st.write("The application looked for:")

    for path in MODEL_CANDIDATES:
        st.code(str(path))

    st.warning(
        "Put insurance_fraud_gradient_boosting.pkl in the same "
        "folder as app.py OR inside a MODELS folder."
    )

    st.stop()


try:

    model = load_model(str(model_path))

except Exception as e:

    st.error("❌ Could not load the trained model.")

    st.code(str(e))

    st.warning(
        "This can happen when the model was created with a "
        "different scikit-learn version."
    )

    st.stop()


# ============================================================
# GET PIPELINE COMPONENTS
# ============================================================

try:

    preprocessor = model.named_steps["preprocessor"]

    gb_classifier = model.named_steps["classifier"]

except Exception as e:

    st.error(
        "❌ The saved model is not in the expected "
        "scikit-learn Pipeline format."
    )

    st.write("Expected pipeline steps:")

    st.code("preprocessor → classifier")

    st.write("Actual model:", model)

    st.write("Error:", e)

    st.stop()


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = None
feature_names = None

if SHAP_AVAILABLE:

    try:

        explainer = shap.TreeExplainer(gb_classifier)

        try:

            feature_names = (
                preprocessor.get_feature_names_out()
            )

        except Exception:

            feature_names = None

    except Exception as e:

        SHAP_AVAILABLE = False

        st.warning(
            "SHAP could not be initialized. "
            "The prediction system will still work."
        )


# ============================================================
# LLM FUNCTION
# ============================================================

def generate_llm_explanation(
    fraud_probability,
    risk_category,
    prediction,
    shap_factors
):

    if not OPENAI_AVAILABLE:

        return (
            "The OpenAI package is not installed. "
            "Add 'openai' to requirements.txt and redeploy."
        )

    # Streamlit Secrets
    api_key = None

    try:

        api_key = st.secrets["OPENAI_API_KEY"]

    except Exception:
        pass

    # Optional environment variable fallback
    if not api_key:

        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:

        return (
            "OpenAI API key is not configured. "
            "Add OPENAI_API_KEY to Streamlit Secrets."
        )

    client = OpenAI(api_key=api_key)

    if shap_factors is None or len(shap_factors) == 0:

        factors_text = "No SHAP factors were available."

    else:

        factor_lines = []

        for _, row in shap_factors.iterrows():

            feature = str(row["Feature"])

            value = float(row["SHAP Value"])

            direction = (
                "increases"
                if value > 0
                else "decreases"
            )

            factor_lines.append(
                f"{feature}: SHAP={value:.4f} "
                f"({direction} fraud-risk score)"
            )

        factors_text = "\n".join(factor_lines)

    prompt = f"""
You are an insurance fraud risk analyst.

Explain the machine-learning assessment of an insurance claim
to a non-technical investigator.

Model prediction:
{prediction}

Fraud probability:
{fraud_probability:.4f}

Risk category:
{risk_category}

Top SHAP factors:
{factors_text}

Write a concise professional explanation.

Your explanation must:
1. State the overall risk level.
2. Explain the most important factors increasing risk.
3. Explain the most important factors decreasing risk.
4. Give a practical investigator-oriented conclusion.

Important safety rules:
- Do NOT say that the customer committed fraud.
- Do NOT treat SHAP values as proof of fraud.
- Do NOT claim that a feature causes fraud.
- Clearly describe this as a machine-learning risk assessment.
- Keep the explanation to approximately 120-180 words.
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        return response.output_text

    except Exception as e:

        return f"Unable to generate AI explanation: {e}"


# ============================================================
# DATA UPLOAD
# ============================================================

st.header("📂 Upload Insurance Claims")

uploaded_file = st.file_uploader(
    "Upload your insurance claims CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload the CSV dataset to start fraud-risk prediction."
    )

    st.stop()


# ============================================================
# READ DATA
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("❌ Could not read the CSV file.")

    st.write("Error:", e)

    st.stop()


st.success(
    f"Successfully loaded {len(df):,} claims."
)


# ============================================================
# DATA PREVIEW
# ============================================================

st.header("📊 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# DATA QUALITY
# ============================================================

st.header("🔎 Data Quality")

quality_df = pd.DataFrame(
    {
        "Feature": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Data Type": df.dtypes.astype(str).values
    }
)

st.dataframe(
    quality_df,
    use_container_width=True
)


# ============================================================
# REMOVE TARGET COLUMN IF PRESENT
# ============================================================

prediction_input = df.copy()

if "FraudFlag" in prediction_input.columns:

    prediction_input = prediction_input.drop(
        columns=["FraudFlag"]
    )


# Remove previously generated application columns
prediction_input = prediction_input.drop(
    columns=[
        "Fraud Probability",
        "Fraud Prediction",
        "Risk Category"
    ],
    errors="ignore"
)


# ============================================================
# PREDICTION
# ============================================================

st.header("🚨 Fraud Risk Prediction")

try:

    probabilities = model.predict_proba(
        prediction_input
    )[:, 1]

except Exception as e:

    st.error("❌ Prediction failed.")

    st.error(
        "The uploaded CSV does not match the input "
        "features expected by the trained model."
    )

    st.write("Prediction error:")

    st.code(str(e))

    st.stop()


# ============================================================
# ADD PREDICTIONS
# ============================================================

df["Fraud Probability"] = probabilities

df["Fraud Prediction"] = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def risk_category(probability):

    if probability < 0.10:

        return "Low Risk"

    elif probability < 0.30:

        return "Medium Risk"

    else:

        return "High Risk"


df["Risk Category"] = [
    risk_category(probability)
    for probability in probabilities
]


# ============================================================
# RISK SUMMARY
# ============================================================

st.subheader("Risk Summary")

risk_counts = df["Risk Category"].value_counts()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🟢 Low Risk",
        risk_counts.get("Low Risk", 0)
    )

with col2:

    st.metric(
        "🟡 Medium Risk",
        risk_counts.get("Medium Risk", 0)
    )

with col3:

    st.metric(
        "🔴 High Risk",
        risk_counts.get("High Risk", 0)
    )


# ============================================================
# RISK FILTER
# ============================================================

st.header("🚨 Risk-Based Claim Filtering")

selected_risk = st.selectbox(
    "Select risk category",
    [
        "All",
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ]
)


if selected_risk == "All":

    filtered_df = df

else:

    filtered_df = df[
        df["Risk Category"] == selected_risk
    ]


st.write(
    f"Showing **{len(filtered_df):,} claims**"
)

st.dataframe(
    filtered_df,
    use_container_width=True
)


# ============================================================
# INDIVIDUAL CLAIM ANALYSIS
# ============================================================

st.header("🔍 Individual Claim Analysis")

claim_number = st.number_input(
    "Select claim row",
    min_value=0,
    max_value=max(len(df) - 1, 0),
    value=0,
    step=1
)

claim_number = int(claim_number)

selected_claim = df.iloc[claim_number]


# ============================================================
# CLAIM RISK
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Fraud Probability",
        f"{selected_claim['Fraud Probability']:.2%}"
    )

with col2:

    st.metric(
        "Risk Category",
        selected_claim["Risk Category"]
    )

with col3:

    prediction_text = (
        "FRAUD"
        if selected_claim["Fraud Prediction"] == 1
        else "NOT FRAUD"
    )

    st.metric(
        "Model Prediction",
        prediction_text
    )


# ============================================================
# CLAIM INFORMATION
# ============================================================

st.subheader("Claim Information")

claim_information = (
    selected_claim
    .drop(
        labels=[
            "Fraud Probability",
            "Fraud Prediction",
            "Risk Category"
        ],
        errors="ignore"
    )
    .to_frame("Value")
)

st.dataframe(
    claim_information,
    use_container_width=True
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.subheader("🧠 SHAP Explanation")

top_factors = pd.DataFrame()

if SHAP_AVAILABLE and explainer is not None:

    try:

        claim_input = (
            prediction_input
            .iloc[[claim_number]]
        )

        claim_transformed = (
            preprocessor.transform(
                claim_input
            )
        )

        # Convert sparse matrix if required
        if hasattr(
            claim_transformed,
            "toarray"
        ):

            claim_transformed = (
                claim_transformed.toarray()
            )

        claim_transformed = np.asarray(
            claim_transformed
        )

        # Get SHAP values
        raw_shap_values = (
            explainer.shap_values(
                claim_transformed
            )
        )

        # Handle different SHAP formats
        if isinstance(
            raw_shap_values,
            list
        ):

            # For binary classification, class 1
            # is the fraud class.
            if len(raw_shap_values) > 1:

                claim_values = np.asarray(
                    raw_shap_values[1]
                )[0]

            else:

                claim_values = np.asarray(
                    raw_shap_values[0]
                )[0]

        else:

            raw_shap_values = np.asarray(
                raw_shap_values
            )

            if raw_shap_values.ndim == 3:

                # Possible shape:
                # samples x features x classes

                if raw_shap_values.shape[-1] > 1:

                    claim_values = (
                        raw_shap_values[0, :, 1]
                    )

                else:

                    claim_values = (
                        raw_shap_values[0, :, 0]
                    )

            elif raw_shap_values.ndim == 2:

                claim_values = (
                    raw_shap_values[0]
                )

            else:

                claim_values = (
                    raw_shap_values.flatten()
                )

        claim_values = np.asarray(
            claim_values
        ).flatten()

        # Get feature names
        if feature_names is None:

            feature_names = np.array(
                [
                    f"Feature_{i}"
                    for i in range(
                        len(claim_values)
                    )
                ]
            )

        feature_names = np.asarray(
            feature_names
        ).flatten()

        # Make sure lengths match
        n_features = min(
            len(feature_names),
            len(claim_values)
        )

        claim_explanation = pd.DataFrame(
            {
                "Feature":
                    feature_names[:n_features],

                "SHAP Value":
                    claim_values[:n_features]
            }
        )

        claim_explanation[
            "Absolute SHAP"
        ] = np.abs(
            claim_explanation["SHAP Value"]
        )

        # Top 10 factors
        top_factors = (
            claim_explanation
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
            .head(10)
            .sort_values(
                "SHAP Value"
            )
        )

        st.write(
            "### Top 10 Contributing Factors"
        )

        display_table = top_factors[
            [
                "Feature",
                "SHAP Value"
            ]
        ].copy()

        display_table[
            "SHAP Value"
        ] = display_table[
            "SHAP Value"
        ].round(4)

        st.dataframe(
            display_table,
            use_container_width=True
        )

        st.write(
            "### Feature Contributions"
        )

        chart_data = (
            top_factors
            .set_index("Feature")[
                "SHAP Value"
            ]
        )

        st.bar_chart(chart_data)

        st.info(
            "Positive SHAP values push the prediction "
            "toward higher fraud risk. Negative SHAP values "
            "push the prediction toward lower fraud risk."
        )

    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated "
            "for this claim."
        )

        st.code(str(e))

else:

    st.info(
        "SHAP is not available. Install the SHAP package "
        "to enable individual claim explanations."
    )


# ============================================================
# AI-GENERATED EXPLANATION
# ============================================================

st.subheader("🤖 AI-Generated Risk Explanation")

st.info(
    "The Gradient Boosting model makes the fraud-risk "
    "prediction. SHAP explains the model behavior. "
    "The AI only converts these results into "
    "investigator-friendly language."
)


if len(top_factors) > 0:

    if st.button(
        "🤖 Generate AI Explanation",
        type="primary"
    ):

        with st.spinner(
            "Generating investigator-friendly explanation..."
        ):

            explanation = (
                generate_llm_explanation(
                    fraud_probability=float(
                        selected_claim[
                            "Fraud Probability"
                        ]
                    ),
                    risk_category=str(
                        selected_claim[
                            "Risk Category"
                        ]
                    ),
                    prediction=prediction_text,
                    shap_factors=top_factors
                )
            )

        st.success(
            "AI explanation generated successfully!"
        )

        st.markdown(explanation)

else:

    st.warning(
        "Generate the SHAP explanation first. "
        "The AI explanation requires SHAP factors."
    )


# ============================================================
# INVESTIGATOR SUMMARY
# ============================================================

st.header("🕵️ Investigator Summary")

probability = float(
    selected_claim["Fraud Probability"]
)

risk = str(
    selected_claim["Risk Category"]
)

prediction = (
    "Fraud"
    if selected_claim["Fraud Prediction"] == 1
    else "Not Fraud"
)


if risk == "High Risk":

    recommendation = (
        "Prioritize this claim for manual investigation."
    )

elif risk == "Medium Risk":

    recommendation = (
        "Consider additional verification before approval."
    )

else:

    recommendation = (
        "No immediate fraud investigation is indicated "
        "by the model."
    )


st.write(
    f"""
**Model Prediction:** {prediction}

**Fraud Probability:** {probability:.2%}

**Risk Category:** {risk}

**Recommended Action:** {recommendation}
"""
)


# ============================================================
# DOWNLOAD
# ============================================================

st.header("⬇️ Download Scored Claims")

csv_output = df.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Scored Claims CSV",
    data=csv_output,
    file_name="scored_insurance_claims.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Insurance Fraud Detection | "
    "Gradient Boosting + SHAP + LLM Explainability"
)

