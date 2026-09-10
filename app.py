import streamlit as st
from pathlib import Path
import pandas as pd

from src.model import train_model, predict_one, predict_batch

st.set_page_config(page_title="Loan Intelligence", layout="wide")

st.markdown("""
<style>
    .stApp { background: #ffffff; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    .main-title { font-size: 2.4rem; font-weight: 700; color: #172033; margin-bottom: 0.2rem; }
    .sub-title { color: #667085; margin-bottom: 1.5rem; }
    .section-card {
        background: #f8fafc;
        border: 1px solid #e6eaf0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e6eaf0;
        border-radius: 12px;
        padding: 0.8rem;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #ffffff;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: #1f6feb;
        color: #ffffff;
        border: 0;
        border-radius: 9px;
        font-weight: 600;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #1558b0;
        color: #ffffff;
    }
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
    return float((value_range[0] + value_range[1]) / 2)


tab1, tab2 = st.tabs(["Single Applicant", "CSV Analysis"])

with tab1:
    st.subheader("Applicant profile")
    st.caption("Select a realistic range for each numeric field. The model uses the midpoint of the selected range.")

    with st.form("loan_form"):
        left, right = st.columns(2)

        with left:
            age_range = st.slider(
                "Age",
                min_value=18, max_value=80, value=(25, 35), step=1,
                help="The model's preprocessing accepts applicant ages up to 80."
            )
            income_range = st.slider(
                "Annual Income ($)",
                min_value=8000, max_value=720000, value=(40000, 70000), step=5000,
                help="Income range aligned with the dataset scale."
            )
            employment_range = st.slider(
                "Employment Experience (years)",
                min_value=0, max_value=60, value=(3, 8), step=1,
                help="The model's preprocessing accepts employment experience up to 60 years."
            )
            loan_amount_range = st.slider(
                "Loan Amount ($)",
                min_value=500, max_value=35000, value=(5000, 15000), step=500,
                help="Loan amount range aligned with the dataset scale."
            )
            interest_range = st.slider(
                "Interest Rate (%)",
                min_value=5.0, max_value=20.0, value=(8.0, 14.0), step=0.1,
                help="Interest-rate range aligned with the dataset scale."
            )
            loan_income_range = st.slider(
                "Loan Percent of Income",
                min_value=0.01, max_value=0.83, value=(0.15, 0.30), step=0.01,
                help="Loan amount as a fraction of annual income."
            )

        with right:
            credit_history_range = st.slider(
                "Credit History Length (years)",
                min_value=2, max_value=30, value=(3, 10), step=1,
                help="Credit-history range aligned with the dataset scale."
            )
            credit_score_range = st.slider(
                "Credit Score",
                min_value=390, max_value=850, value=(600, 750), step=5,
                help="Credit-score range aligned with the dataset scale."
            )
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
