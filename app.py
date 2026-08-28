import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("🔍 Insurance Claim Fraud Detection System")

st.markdown("""
This application uses a **Gradient Boosting machine learning model**
to estimate insurance claim fraud risk.

The system provides:

- Fraud probability
- Low / Medium / High risk classification
- SHAP-based feature explanations
- Individual claim analysis
- Scored claim download
- Investigator-friendly AI explanation
""")

st.divider()

# ============================================================
# CONSTANTS
# ============================================================

THRESHOLD = 0.10

MODEL_PATH = "MODELS/insurance_fraud_gradient_boosting.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error("❌ Could not load the trained model.")

    st.write("Model loading error:", e)

    st.info(
        "Make sure the file exists at: "
        "models/insurance_fraud_gradient_boosting.pkl"
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

    st.write("Error:", e)

    st.stop()


# ============================================================
# SHAP EXPLAINER
# ============================================================

try:

    explainer = shap.TreeExplainer(
        gb_classifier
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    shap_available = True

except Exception as e:

    shap_available = False

    st.warning(
        "SHAP explanation is currently unavailable."
    )

    st.write("SHAP error:", e)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📁 Upload Claims")

uploaded_file = st.sidebar.file_uploader(
    "Upload insurance claims CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Fraud threshold:** {THRESHOLD}"
)

st.sidebar.write(
    "**Model:** Gradient Boosting"
)

st.sidebar.write(
    "**ROC-AUC:** 0.8445"
)


# ============================================================
# IF NO FILE UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload your insurance claims CSV file "
        "from the sidebar to begin."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("❌ Could not read the CSV file.")

    st.write("Error:", e)

    st.stop()


# ============================================================
# DATA OVERVIEW
# ============================================================

st.header("📊 Data Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Number of Claims",
        len(df)
    )

with col2:

    st.metric(
        "Number of Features",
        len(df.columns)
    )

with col3:

    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


st.subheader("Preview")

st.dataframe(
    df.head(100),
    use_container_width=True
)


# ============================================================
# DATA QUALITY
# ============================================================

st.header("🔎 Data Quality")

quality_df = pd.DataFrame({

    "Feature": df.columns,

    "Missing Values": (
        df.isnull().sum().values
    ),

    "Data Type": (
        df.dtypes.astype(str).values
    )

})

st.dataframe(
    quality_df,
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

st.header("🚨 Fraud Risk Prediction")

try:

    probabilities = (
        model.predict_proba(df)[:, 1]
    )

except Exception as e:

    st.error(
        "❌ Prediction failed."
    )

    st.error(
        "The uploaded CSV does not match the "
        "input features expected by the trained model."
    )

    st.write("Prediction error:", e)

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

risk_counts = (
    df["Risk Category"]
    .value_counts()
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🟢 Low Risk",
        risk_counts.get(
            "Low Risk",
            0
        )
    )

with col2:

    st.metric(
        "🟡 Medium Risk",
        risk_counts.get(
            "Medium Risk",
            0
        )
    )

with col3:

    st.metric(
        "🔴 High Risk",
        risk_counts.get(
            "High Risk",
            0
        )
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
    f"Showing **{len(filtered_df)} claims**"
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
    max_value=len(df) - 1,
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

if shap_available:

    try:

        # Remove application-generated columns
        claim_input = (
            df.drop(
                columns=[
                    "Fraud Probability",
                    "Fraud Prediction",
                    "Risk Category"
                ],
                errors="ignore"
            )
            .iloc[[claim_number]]
        )

        # Apply original preprocessing
        claim_transformed = (
            preprocessor.transform(
                claim_input
            )
        )

        # Calculate SHAP values
        claim_shap_values = (
            explainer.shap_values(
                claim_transformed
            )
        )

        # Handle different SHAP output formats
        if isinstance(
            claim_shap_values,
            list
        ):

            claim_values = (
                claim_shap_values[0]
            )

        else:

            claim_values = (
                claim_shap_values[0]
            )

        claim_values = np.asarray(
            claim_values
        ).flatten()

        # Create SHAP dataframe
        claim_explanation = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": claim_values

        })

        claim_explanation[
            "Absolute SHAP"
        ] = np.abs(
            claim_explanation[
                "SHAP Value"
            ]
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

        # Display table
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

        # SHAP chart
        st.write(
            "### Feature Contributions"
        )

        chart_data = (
            top_factors
            .set_index("Feature")[
                "SHAP Value"
            ]
        )

        st.bar_chart(
            chart_data
        )

        st.info(
            "Positive SHAP values push the prediction "
            "toward higher fraud risk. Negative SHAP "
            "values push the prediction toward lower "
            "fraud risk."
        )

    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated "
            "for this claim."
        )

        st.write(
            "SHAP Error:",
            e
        )


# ============================================================
# INVESTIGATOR SUMMARY
# ============================================================

st.header("🕵️ Investigator Summary")

probability = float(
    selected_claim["Fraud Probability"]
)

risk = selected_claim[
    "Risk Category"
]

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
# GENERATIVE AI EXPLANATION
# ============================================================

st.subheader("🤖 AI Investigator Explanation")

st.info(
    "The AI explanation is an interpretation layer. "
    "The Gradient Boosting model makes the fraud-risk "
    "prediction; the AI does not make the fraud decision."
)


# Prepare top factors for the explanation
if shap_available and "top_factors" in locals():

    positive_factors = (
        top_factors[
            top_factors["SHAP Value"] > 0
        ]
        .sort_values(
            "SHAP Value",
            ascending=False
        )
        .head(5)
    )

    negative_factors = (
        top_factors[
            top_factors["SHAP Value"] < 0
        ]
        .sort_values(
            "SHAP Value"
        )
        .head(5)
    )

else:

    positive_factors = pd.DataFrame()

    negative_factors = pd.DataFrame()


st.write("### Investigator-Friendly Explanation")

if probability >= 0.30:

    st.warning(
        f"This claim has been classified as **High Risk** "
        f"with a model-estimated fraud probability of "
        f"**{probability:.2%}**. "
        f"The claim should be prioritized for further "
        f"manual investigation."
    )

elif probability >= 0.10:

    st.warning(
        f"This claim has been classified as **Medium Risk** "
        f"with a model-estimated fraud probability of "
        f"**{probability:.2%}**. "
        f"Additional verification may be appropriate."
    )

else:

    st.success(
        f"This claim has been classified as **Low Risk** "
        f"with a model-estimated fraud probability of "
        f"**{probability:.2%}**. "
        f"The model does not indicate immediate fraud risk."
    )


# Positive factors
if len(positive_factors) > 0:

    st.write(
        "**Factors increasing the model's fraud-risk score:**"
    )

    for _, row in positive_factors.iterrows():

        st.write(
            f"- {row['Feature']} "
            f"(SHAP: +{row['SHAP Value']:.4f})"
        )


# Negative factors
if len(negative_factors) > 0:

    st.write(
        "**Factors decreasing the model's fraud-risk score:**"
    )

    for _, row in negative_factors.iterrows():

        st.write(
            f"- {row['Feature']} "
            f"(SHAP: {row['SHAP Value']:.4f})"
        )


st.caption(
    "Note: SHAP factors describe model behavior and "
    "should not be interpreted as proof of fraud or causation."
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
    "Gradient Boosting + SHAP Explainability"
)
