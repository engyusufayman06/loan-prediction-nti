import streamlit as st
from pathlib import Path

from src.model import train_model, predict_one

# ---------------------------------------------------------
# Simple Loan Intelligence Demo
# ---------------------------------------------------------

st.set_page_config(
    page_title="Loan Intelligence",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Loan Intelligence")
st.caption("AI-Powered Credit Decision Demo | NTI Machine Learning Track")

DATA_PATH = Path("loan_data.csv")


@st.cache_resource
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "loan_data.csv was not found. Put it in the same folder as app.py."
        )
    bundle, metrics = train_model(str(DATA_PATH))
    return bundle, metrics


st.info(
    "Enter an applicant's information below and let the trained XGBoost model "
    "make a prediction."
)

with st.form("loan_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=18.0, max_value=80.0, value=30.0, step=1.0
        )
        income = st.number_input(
            "Annual Income", min_value=0.0, value=50000.0, step=1000.0
        )
        home = st.selectbox(
            "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )
        employment = st.number_input(
            "Employment Experience (years)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=1.0,
        )
        education = st.selectbox(
            "Education",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
        )
        gender = st.selectbox("Gender", ["male", "female"])
        defaults = st.selectbox("Previous Loan Defaults", ["No", "Yes"])

    with col2:
        intent = st.selectbox(
            "Loan Intent",
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
            ],
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0, value=10000.0, step=500.0
        )
        interest_rate = st.number_input(
            "Loan Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
        )
        loan_percent_income = st.number_input(
            "Loan Percent of Income",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
        )
        credit_history = st.number_input(
            "Credit History Length (years)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=1.0,
        )
        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0,
            max_value=900.0,
            value=650.0,
            step=1.0,
        )

    submitted = st.form_submit_button(
        "🔮 Predict Loan Decision", use_container_width=True
    )


if submitted:
    try:
        with st.spinner("AI model is analyzing the applicant..."):
            bundle, metrics = load_model()

            applicant = {
                "person_age": float(age),
                "person_income": float(income),
                "person_home_ownership": home,
                "person_emp_exp": float(employment),
                "loan_intent": intent,
                "loan_amnt": float(loan_amount),
                "loan_int_rate": float(interest_rate),
                "loan_percent_income": float(loan_percent_income),
                "cb_person_cred_hist_length": float(credit_history),
                "credit_score": float(credit_score),
                "previous_loan_defaults_on_file": defaults,
                "person_gender": gender,
                "person_education": education,
            }

            result = predict_one(bundle, applicant)

        st.divider()

        if result["prediction"] == 1:
            st.success("### ✅ LOAN APPROVED")
        else:
            st.error("### ❌ LOAN REJECTED")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Approval Score", f'{result["approval_probability"] * 100:.2f}%')
        with c2:
            st.metric("Rejection Score", f'{result["rejection_probability"] * 100:.2f}%')

        with st.expander("Model Information"):
            st.write("**Model:** XGBoost")
            st.write(f'**Test Accuracy:** {metrics["accuracy"] * 100:.2f}%')
            st.caption(
                "These probabilities are model scores, not guaranteed real-world approval odds."
            )

    except Exception as e:
        st.error("The demo could not run.")
        st.exception(e)
