from pathlib import Path

import streamlit as st

from src.model import predict_one, train_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "loan_data.csv"

st.set_page_config(page_title="Loan Prediction", page_icon="💳", layout="centered")

st.title("💳 Loan Prediction")
st.caption("Simple XGBoost model demo")
st.write("Enter the applicant information and test the trained model.")


@st.cache_resource(show_spinner="Training model...")
def load_model():
    return train_model(DATA_PATH)


try:
    bundle, _ = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.divider()
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
    income = st.number_input("Annual Income", min_value=0.0, value=60000.0, step=1000.0)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=60, value=5, step=1)
    intent = st.selectbox(
        "Loan Intent",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0, step=0.1)

with col2:
    loan_percent_income = st.number_input(
        "Loan / Income Ratio", min_value=0.0, max_value=1.0, value=0.17, step=0.01
    )
    credit_history = st.number_input(
        "Credit History Length (years)", min_value=0.0, max_value=30.0, value=5.0, step=1.0
    )
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=1)
    previous_default = st.selectbox("Previous Loan Defaults", ["No", "Yes"])
    gender = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox(
        "Education",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    )

st.divider()

if st.button("🔮 Predict Loan Status", type="primary", use_container_width=True):
    applicant = {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_exp": emp_exp,
        "loan_intent": intent,
        "loan_amnt": loan_amount,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": credit_history,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_default,
        "person_gender": gender,
        "person_education": education,
    }

    try:
        result = predict_one(bundle, applicant)
        if result["prediction"] == 1:
            st.success("### ✅ Loan Status: Approved")
        else:
            st.error("### ❌ Loan Status: Rejected")

        c1, c2 = st.columns(2)
        c1.metric("Approval Probability", f"{result['approval_probability']:.2%}")
        c2.metric("Rejection Probability", f"{result['rejection_probability']:.2%}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("XGBoost • Test Accuracy: 93.47% • Educational project — not a real lending decision system.")
