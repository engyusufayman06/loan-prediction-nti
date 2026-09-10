import streamlit as st
from pathlib import Path
import pandas as pd

from src.model import train_model, predict_one, predict_batch

st.set_page_config(page_title="Loan Intelligence", layout="wide")

st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    .main-title {font-size: 2.4rem; font-weight: 700; color: #1f2937; margin-bottom: 0.2rem;}
    .sub-title {color: #6b7280; margin-bottom: 1.5rem;}
    .result-box {padding: 1.2rem; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Loan Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered credit decision demo</div>', unsafe_allow_html=True)

DATA_PATH = Path("loan_data.csv")

@st.cache_resource(show_spinner="Training model once... this can take a little while on first run.")
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("loan_data.csv was not found in the project folder.")
    return train_model(DATA_PATH)


def midpoint(value_range):
    return float(sum(value_range) / 2)


tab1, tab2 = st.tabs(["Single Applicant", "CSV Analysis"])

with tab1:
    st.subheader("Applicant profile")
    st.caption("Choose ranges instead of entering exact numbers. The model uses the midpoint of each selected range.")

    with st.form("loan_form"):
        left, right = st.columns(2)

        with left:
            age_range = st.slider("Age", 18, 80, (25, 35))
            income_range = st.slider("Annual Income", 10000, 200000, (40000, 70000), step=5000)
            employment_range = st.slider("Employment Experience (years)", 0, 40, (3, 8))
            loan_amount_range = st.slider("Loan Amount", 1000, 50000, (5000, 15000), step=1000)
            interest_range = st.slider("Interest Rate (%)", 1.0, 30.0, (8.0, 14.0), step=0.5)
            loan_income_range = st.slider("Loan Percent of Income", 0.01, 0.80, (0.15, 0.30), step=0.01)

        with right:
            credit_history_range = st.slider("Credit History Length (years)", 0, 30, (3, 10))
            credit_score_range = st.slider("Credit Score", 300, 850, (600, 750), step=10)
            home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
            gender = st.selectbox("Gender", ["male", "female"])
            defaults = st.selectbox("Previous Loan Defaults", ["No", "Yes"])
            intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

        submitted = st.form_submit_button("Predict Decision", use_container_width=True)

    if submitted:
        try:
            bundle, metrics = load_model()
            applicant = {
                "person_age": midpoint(age_range),
                "person_income": midpoint(income_range),
                "person_home_ownership": home,
                "person_emp_exp": midpoint(employment_range),
                "loan_intent": intent,
                "loan_amnt": midpoint(loan_amount_range),
                "loan_int_rate": midpoint(interest_range),
                "loan_percent_income": midpoint(loan_income_range),
                "cb_person_cred_hist_length": midpoint(credit_history_range),
                "credit_score": midpoint(credit_score_range),
                "previous_loan_defaults_on_file": defaults,
                "person_gender": gender,
                "person_education": education,
            }
            result = predict_one(bundle, applicant)

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Decision", result["label"])
            c2.metric("Approval Score", f'{result["approval_probability"] * 100:.1f}%')
            c3.metric("Rejection Score", f'{result["rejection_probability"] * 100:.1f}%')

            st.subheader("Decision confidence")
            chart = pd.DataFrame({
                "Score": [result["approval_probability"], result["rejection_probability"]]
            }, index=["Approval", "Rejection"])
            st.bar_chart(chart, horizontal=True)

            with st.expander("Model details"):
                st.write("Model: XGBoost")
                st.write(f'Test accuracy: {metrics["accuracy"] * 100:.2f}%')
                st.write(f'Encoded features: {metrics["feature_count"]}')
                st.caption("Probabilities are model scores, not guaranteed real-world approval odds.")
        except Exception as e:
            st.error("The prediction could not be completed.")
            st.exception(e)

with tab2:
    st.subheader("Upload a CSV for batch analysis")
    st.caption("Upload a CSV containing the same applicant fields used by the model. Results can be downloaded after analysis.")

    sample_path = Path("demo_test.csv")
    if sample_path.exists():
        st.download_button(
            "Download Test CSV",
            data=sample_path.read_bytes(),
            file_name="loan_prediction_test.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Analyze CSV", use_container_width=True):
                bundle, metrics = load_model()
                results = predict_batch(bundle, df)

                st.success(f"Analysis complete: {len(results):,} applicants processed.")

                approved = int((results["prediction"] == 1).sum())
                rejected = int((results["prediction"] == 0).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Applicants", len(results))
                c2.metric("Approved", approved)
                c3.metric("Rejected", rejected)

                st.subheader("Prediction distribution")
                distribution = pd.DataFrame({"Applicants": [approved, rejected]}, index=["Approved", "Rejected"])
                st.bar_chart(distribution)

                st.subheader("Approval score distribution")
                st.line_chart(results["approval_probability"].reset_index(drop=True))

                st.subheader("Results")
                st.dataframe(results, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Prediction Results",
                    data=csv,
                    file_name="loan_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error("Could not analyze this CSV. Make sure it contains the required model input columns.")
            st.exception(e)
