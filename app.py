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

        st.write(
            f"### Claim Row: {claim_number}"
        )

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

        st.subheader("Claim Information")

        st.dataframe(
            selected_claim.to_frame("Value"),
            use_container_width=True
        )

        # -----------------------------------
        # DOWNLOAD
        # -----------------------------------

        st.header("⬇️ Download Results")

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Scored Claims CSV",
            data=csv,
            file_name="scored_insurance_claims.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            "The uploaded CSV does not match the format expected "
            "by the trained model."
        )

        st.write("Error:", e)

else:

    st.info(
        "Please upload an insurance claims CSV file using "
        "the sidebar."
    )
