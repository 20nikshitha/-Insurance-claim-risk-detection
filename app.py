import joblib 
import shap
import numpy as np
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🔍 Insurance Claim Fraud Detection")
st.write(
    "Machine learning-based system for identifying potentially "
    "fraudulent and high-risk insurance claims."
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "MODELS/insurance_fraud_gradient_boosting.pkl"
    )

model = load_model()
gb_classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

explainer = shap.TreeExplainer(gb_classifier)

feature_names = preprocessor.get_feature_names_out()
# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.header("Upload Claims Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload insurance claims CSV",
    type=["csv"]
)

# -----------------------------------
# MAIN APPLICATION
# -----------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.header("📊 Data Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of Claims", len(df))

    with col2:
        st.metric("Number of Columns", len(df.columns))

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.subheader("Uploaded Data")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    # -----------------------------------
    # DATA QUALITY
    # -----------------------------------

    st.header("🔎 Data Quality")

    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        quality,
        use_container_width=True
    )

    # -----------------------------------
    # PREDICTION
    # -----------------------------------

    st.header("🚨 Fraud Risk Prediction")

    try:

        probabilities = model.predict_proba(df)[:, 1]

        df["Fraud Probability"] = probabilities

        # Selected threshold from our project
        threshold = 0.10

        df["Fraud Prediction"] = (
            probabilities >= threshold
        ).astype(int)

        # Risk categories
        def risk_category(probability):

            if probability < 0.10:
                return "Low Risk"

            elif probability < 0.30:
                return "Medium Risk"

            else:
                return "High Risk"

        df["Risk Category"] = [
            risk_category(p)
            for p in probabilities
        ]

        # -----------------------------------
        # RISK SUMMARY
        # -----------------------------------

        st.subheader("Risk Summary")

        risk_counts = (
            df["Risk Category"]
            .value_counts()
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Low Risk",
                risk_counts.get("Low Risk", 0)
            )

        with c2:
            st.metric(
                "Medium Risk",
                risk_counts.get("Medium Risk", 0)
            )

        with c3:
            st.metric(
                "High Risk",
                risk_counts.get("High Risk", 0)
            )

        # -----------------------------------
        # FILTER CLAIMS
        # -----------------------------------

        st.subheader("Filter Claims")

        selected_risk = st.selectbox(
            "Select Risk Category",
            ["All", "High Risk", "Medium Risk", "Low Risk"]
        )

        if selected_risk == "All":

            filtered_df = df

        else:

            filtered_df = df[
                df["Risk Category"] == selected_risk
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        # -----------------------------------
# INDIVIDUAL CLAIM
# -----------------------------------

st.header("🔍 Individual Claim Analysis")

claim_number = st.number_input(
    "Select claim row",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

selected_claim = df.iloc[claim_number]

st.write(f"### Claim Row: {claim_number}")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Fraud Probability",
        f"{selected_claim['Fraud Probability']:.2%}"
    )

with c2:
    st.metric(
        "Risk Category",
        selected_claim["Risk Category"]
    )

# -----------------------------------
# CLAIM INFORMATION
# -----------------------------------

st.subheader("Claim Information")

st.dataframe(
    selected_claim.to_frame("Value"),
    use_container_width=True
)

# -----------------------------------
# SHAP EXPLANATION
# -----------------------------------

st.subheader("🧠 SHAP Contributing Factors")

try:

    # Select original claim before prediction columns were added
    claim_input = df.drop(
        columns=[
            "Fraud Probability",
            "Fraud Prediction",
            "Risk Category"
        ],
        errors="ignore"
    ).iloc[[claim_number]]

    # Transform claim using the same preprocessing pipeline
    claim_transformed = preprocessor.transform(claim_input)

    # Calculate SHAP values
    claim_shap_values = explainer.shap_values(
        claim_transformed
    )

    # Get SHAP values for this claim
    if isinstance(claim_shap_values, list):
        claim_values = claim_shap_values[0]
    else:
        claim_values = claim_shap_values[0]

    # Create explanation dataframe
    claim_explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": claim_values
    })

    claim_explanation["Absolute SHAP"] = (
        np.abs(claim_explanation["SHAP Value"])
    )

    # Top 10 factors
    top_factors = (
        claim_explanation
        .sort_values("Absolute SHAP", ascending=False)
        .head(10)
        .sort_values("SHAP Value")
    )

    # Display table
    st.write("### Top 10 Factors")

    display_table = top_factors[
        ["Feature", "SHAP Value"]
    ].copy()

    display_table["SHAP Value"] = (
        display_table["SHAP Value"]
        .round(4)
    )

    st.dataframe(
        display_table,
        use_container_width=True
    )

    # -----------------------------------
    # SHAP BAR CHART
    # -----------------------------------

    st.write("### Feature Contributions")

    chart_data = top_factors.set_index("Feature")[
        "SHAP Value"
    ]

    st.bar_chart(chart_data)

    st.info(
        "Positive SHAP values increase the model's fraud prediction, "
        "while negative SHAP values decrease it."
    )

except Exception as e:

    st.warning(
        "SHAP explanation could not be generated for this claim."
    )

    st.write("Error:", e)
