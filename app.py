import streamlit as st
from pathlib import Path
import pandas as pd

from src.model import train_model, predict_one, predict_batch

st.set_page_config(page_title="Loan Intelligence", page_icon=None, layout="wide")

st.markdown("""
<style>
    .stApp { background: #ffffff; color: #172033; }
    .block-container { max-width: 1200px; padding-top: 1.8rem; padding-bottom: 3rem; }
    .main-title { font-size: 2.6rem; font-weight: 750; color: #172033; letter-spacing: -0.03em; margin-bottom: 0.1rem; }
    .sub-title { color: #667085; font-size: 1rem; margin-bottom: 1.8rem; }
    .panel { background: #f8fafc; border: 1px solid #e5eaf1; border-radius: 16px; padding: 1.2rem 1.3rem; }
    .small-label { color: #667085; font-size: 0.82rem; margin-bottom: 0.2rem; }
    .big-value { color: #172033; font-size: 1.65rem; font-weight: 700; }
    .decision-approved { color: #15803d; font-weight: 750; font-size: 1.8rem; }
    .decision-rejected { color: #dc2626; font-weight: 750; font-size: 1.8rem; }
    div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e5eaf1; border-radius: 14px; padding: 0.8rem 1rem; }
    div[data-baseweb="select"] > div { background: #ffffff; border-color: #d8dee8; }
    .stSlider > div > div > div > div { background: #2563eb; }
    .stButton > button, .stFormSubmitButton > button { background: #2563eb; color: #ffffff; border: 0; border-radius: 10px; font-weight: 650; min-height: 2.7rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background: #1d4ed8; color: #ffffff; }
    .stDownloadButton > button { border-radius: 10px; border: 1px solid #cbd5e1; background: #ffffff; color: #172033; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Loan Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered credit decision and applicant analysis</div>', unsafe_allow_html=True)

DATA_PATH = Path("loan_data.csv")

@st.cache_resource(show_spinner="Preparing the model for the first prediction...")
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("loan_data.csv was not found in the project folder.")
    return train_model(DATA_PATH)


def metric_card(label, value):
    st.markdown(f'<div class="panel"><div class="small-label">{label}</div><div class="big-value">{value}</div></div>', unsafe_allow_html=True)


tab1, tab2 = st.tabs(["Single Applicant", "CSV Analysis"])

with tab1:
    st.subheader("Applicant profile")
    st.caption("Use the sliders to choose one value from each realistic range. The selected value is sent directly to the model.")

    with st.form("loan_form"):
        left, right = st.columns(2, gap="large")

        with left:
            age = st.slider("Age", 18, 80, 28, 1, help="Applicant age. Values above 80 are excluded by the model preprocessing.")
            income = st.slider("Annual Income ($)", 8000, 720000, 60000, 5000, help="Annual income.")
            employment = st.slider("Employment Experience (years)", 0, 60, 6, 1, help="Years of employment experience.")
            loan_amount = st.slider("Loan Amount ($)", 500, 35000, 12000, 500, help="Requested loan amount.")
            interest = st.slider("Interest Rate (%)", 5.0, 20.0, 10.0, 0.1, help="Loan interest rate.")
            loan_income = st.slider("Loan Percent of Income", 0.01, 0.83, 0.20, 0.01, format="%.2f", help="Loan amount as a fraction of annual income.")

        with right:
            credit_history = st.slider("Credit History Length (years)", 2, 30, 7, 1, help="Length of credit history.")
            credit_score = st.slider("Credit Score", 390, 850, 680, 5, help="Credit score.")
            home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
            gender = st.selectbox("Gender", ["male", "female"])
            defaults = st.selectbox("Previous Loan Defaults", ["No", "Yes"])
            intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

        submitted = st.form_submit_button("Predict Loan Decision", use_container_width=True)

    if submitted:
        try:
            bundle, metrics = load_model()
            applicant = {
                "person_age": int(age),
                "person_income": int(income),
                "person_home_ownership": home,
                "person_emp_exp": int(employment),
                "loan_intent": intent,
                "loan_amnt": int(loan_amount),
                "loan_int_rate": float(interest),
                "loan_percent_income": float(loan_income),
                "cb_person_cred_hist_length": int(credit_history),
                "credit_score": int(credit_score),
                "previous_loan_defaults_on_file": defaults,
                "person_gender": gender,
                "person_education": education,
            }
            result = predict_one(bundle, applicant)

            st.divider()
            st.subheader("Prediction result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Decision", result["label"])
            c2.metric("Approval Score", f'{result["approval_probability"] * 100:.1f}%')
            c3.metric("Rejection Score", f'{result["rejection_probability"] * 100:.1f}%')

            chart_col, profile_col = st.columns(2, gap="large")
            with chart_col:
                st.markdown("#### Decision score")
                score_df = pd.DataFrame({"Score": [result["approval_probability"], result["rejection_probability"]]}, index=["Approval", "Rejection"])
                st.bar_chart(score_df, horizontal=True, height=260)

                st.markdown("#### Approval level")
                st.progress(float(result["approval_probability"]))
                st.caption(f'Approval score: {result["approval_probability"] * 100:.1f}%')

            with profile_col:
                st.markdown("#### Applicant snapshot")
                snapshot = pd.DataFrame({
                    "Field": ["Age", "Income", "Loan Amount", "Interest Rate", "Credit Score", "Credit History"],
                    "Selected value": [f"{age} years", f"${income:,.0f}", f"${loan_amount:,.0f}", f"{interest:.1f}%", f"{credit_score}", f"{credit_history} years"],
                })
                st.dataframe(snapshot, hide_index=True, use_container_width=True)

                st.markdown("#### Model summary")
                s1, s2 = st.columns(2)
                s1.metric("Test Accuracy", f'{metrics["accuracy"] * 100:.2f}%')
                s2.metric("Encoded Features", metrics["feature_count"])

            st.caption("The displayed probabilities are model scores from XGBoost, not guaranteed real-world approval odds.")

            with st.expander("Technical model details"):
                st.write("Model: XGBoost")
                st.write(f'Test accuracy: {metrics["accuracy"] * 100:.2f}%')
                st.write(f'Precision: {metrics["precision"] * 100:.2f}%')
                st.write(f'Recall: {metrics["recall"] * 100:.2f}%')
                st.write(f'F1 score: {metrics["f1"] * 100:.2f}%')
        except Exception as e:
            st.error("The prediction could not be completed.")
            st.exception(e)

with tab2:
    st.subheader("CSV analysis dashboard")
    st.caption("Upload applicant records, run the same trained inference pipeline, explore the results, and download the predictions.")

    sample_path = Path("demo_test.csv")
    if sample_path.exists():
        st.download_button("Download Test CSV", data=sample_path.read_bytes(), file_name="loan_prediction_test.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.markdown("#### Dataset preview")
            m1, m2 = st.columns(2)
            m1.metric("Rows", f"{len(df):,}")
            m2.metric("Columns", f"{len(df.columns):,}")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Analyze CSV", use_container_width=True):
                bundle, metrics = load_model()
                results = predict_batch(bundle, df)

                approved = int((results["prediction"] == 1).sum())
                rejected = int((results["prediction"] == 0).sum())
                approval_rate = approved / len(results) if len(results) else 0.0
                avg_approval = float(results["approval_probability"].mean()) if len(results) else 0.0

                st.success(f"Analysis complete: {len(results):,} applicants processed.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Applicants", f"{len(results):,}")
                c2.metric("Approved", f"{approved:,}")
                c3.metric("Rejected", f"{rejected:,}")
                c4.metric("Approval Rate", f"{approval_rate * 100:.1f}%")

                st.divider()
                chart1, chart2 = st.columns(2, gap="large")
                with chart1:
                    st.markdown("#### Prediction distribution")
                    distribution = pd.DataFrame({"Applicants": [approved, rejected]}, index=["Approved", "Rejected"])
                    st.bar_chart(distribution, height=300)

                with chart2:
                    st.markdown("#### Approval score by applicant")
                    score_series = results["approval_probability"].reset_index(drop=True)
                    score_series.index = score_series.index + 1
                    st.line_chart(score_series, height=300)

                st.markdown("#### Approval score bands")
                bands = pd.cut(results["approval_probability"], bins=[0, 0.25, 0.50, 0.75, 1.0], labels=["0–25%", "25–50%", "50–75%", "75–100%"], include_lowest=True)
                band_counts = bands.value_counts().sort_index()
                st.bar_chart(pd.DataFrame({"Applicants": band_counts}))

                if "loan_amnt" in df.columns:
                    scatter = results.copy()
                    source_valid = df.loc[:len(scatter)-1].copy() if len(df) >= len(scatter) else df.copy()
                    if "loan_amnt" in source_valid.columns:
                        scatter = scatter.reset_index(drop=True)
                        scatter["loan_amount"] = source_valid["loan_amnt"].reset_index(drop=True).iloc[:len(scatter)].values
                        scatter["applicant"] = range(1, len(scatter) + 1)
                        st.markdown("#### Loan amount vs approval score")
                        st.scatter_chart(scatter, x="loan_amount", y="approval_probability", height=320)

                st.markdown("#### Prediction results")
                st.dataframe(results, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button("Download Prediction Results", data=csv, file_name="loan_prediction_results.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error("Could not analyze this CSV. Make sure it contains the required model input columns.")
            st.exception(e)
