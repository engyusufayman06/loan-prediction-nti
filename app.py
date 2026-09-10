import streamlit as st
from pathlib import Path
import pandas as pd

from src.model import train_model, predict_one, predict_batch

st.set_page_config(page_title="Loan Intelligence", page_icon=None, layout="wide")

st.markdown("""
<style>
    .stApp { background: #ffffff; color: #18324a; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
    .main-title { font-size: 2.55rem; font-weight: 750; color: #18324a; letter-spacing: -0.03em; margin-bottom: .15rem; }
    .sub-title { color: #6b7b8c; font-size: 1rem; margin-bottom: 1.8rem; }
    .section-note { background: #f5f8fb; border: 1px solid #e1e8ef; border-radius: 12px; padding: .8rem 1rem; color: #526273; }
    .value-pill { display: inline-block; background: #edf4f8; border: 1px solid #d8e6ee; color: #285a76; border-radius: 8px; padding: 3px 9px; font-size: .82rem; font-weight: 650; margin-top: -4px; margin-bottom: 8px; }
    .insight-card { background: #f8fafc; border: 1px solid #e2e8ef; border-radius: 14px; padding: 1rem 1.1rem; min-height: 92px; }
    .insight-label { color: #6b7b8c; font-size: .82rem; }
    .insight-value { color: #18324a; font-size: 1.45rem; font-weight: 750; margin-top: .15rem; }
    div[data-testid="stMetric"] { background: #f7f9fb; border: 1px solid #e2e8ef; border-radius: 13px; padding: .75rem 1rem; }
    div[data-baseweb="select"] > div { background: #ffffff; border-color: #d5dee7; }
    .stButton > button, .stFormSubmitButton > button { background: #2f6f8f; color: #ffffff; border: 0; border-radius: 9px; font-weight: 650; min-height: 2.7rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background: #255b75; color: #ffffff; }
    .stDownloadButton > button { border-radius: 9px; border: 1px solid #cfd9e2; background: #ffffff; color: #29465a; }
    hr { border-color: #e6ebf0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Loan Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered credit decision and model analytics</div>', unsafe_allow_html=True)

DATA_PATH = Path("loan_data.csv")

@st.cache_resource(show_spinner="Preparing the model for the first prediction...")
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("loan_data.csv was not found in the project folder.")
    return train_model(DATA_PATH)


def slider_with_value(label, min_value, max_value, value, step, formatter, help_text):
    selected = st.slider(label, min_value, max_value, value, step, help=help_text)
    st.markdown(f'<div class="value-pill">Selected: {formatter(selected)}</div>', unsafe_allow_html=True)
    return selected


# Benchmark results from the completed model experiments notebook.
BENCHMARK = pd.DataFrame({
    "Accuracy": [0.9328, 0.9201, 0.9171, 0.9168, 0.9008, 0.8858, 0.8819, 0.8652, 0.8640],
    "Precision": [0.8700, 0.8111, 0.7946, 0.8000, 0.7311, 0.6935, 0.6727, 0.6463, 0.6344],
    "Recall": [0.8200, 0.8350, 0.8455, 0.8340, 0.8755, 0.8710, 0.9125, 0.8690, 0.9155],
    "F1 Score": [0.8443, 0.8229, 0.8193, 0.8166, 0.7968, 0.7722, 0.7745, 0.7413, 0.7495],
}, index=[
    "XGBoost", "Random Forest", "HistGradientBoosting", "Extra Trees",
    "Gradient Boosting", "Decision Tree", "SVM", "KNN", "Logistic Regression"
])

FEATURE_IMPORTANCE = pd.DataFrame({
    "Importance": [
        0.805732, 0.031212, 0.020834, 0.020783, 0.018895,
        0.017310, 0.016857, 0.012999, 0.009649, 0.008511,
        0.007039, 0.005432, 0.004760, 0.004585, 0.004117
    ]
}, index=[
    "Previous loan defaults", "Home ownership: RENT", "Home ownership: OWN",
    "Education", "Loan percent of income", "Intent: VENTURE", "Interest rate",
    "Credit history length", "Income", "Intent: EDUCATION", "Intent: HOMEIMPROVEMENT",
    "Intent: PERSONAL", "Employment experience", "Age", "Intent: MEDICAL"
])

CONFUSION_MATRIX = pd.DataFrame(
    [[6753, 245], [360, 1640]],
    index=["Actual Rejected", "Actual Approved"],
    columns=["Predicted Rejected", "Predicted Approved"]
)


tab1, tab2, tab3 = st.tabs(["Single Applicant", "CSV Analysis", "Model Insights"])

with tab1:
    st.subheader("Applicant profile")
    st.markdown('<div class="section-note">Choose an exact value by moving each slider. The selected value is sent directly to the model.</div>', unsafe_allow_html=True)
    st.write("")

    with st.form("loan_form"):
        left, right = st.columns(2, gap="large")

        with left:
            age = slider_with_value("Age", 18, 80, 28, 1, lambda x: f"{x} years", "Applicant age.")
            income = slider_with_value("Annual Income ($)", 8000, 720000, 60000, 1000, lambda x: f"${x:,.0f}", "Annual income.")
            employment = slider_with_value("Employment Experience (years)", 0, 60, 6, 1, lambda x: f"{x} years", "Years of employment experience.")
            loan_amount = slider_with_value("Loan Amount ($)", 500, 35000, 12000, 100, lambda x: f"${x:,.0f}", "Requested loan amount.")
            interest = slider_with_value("Interest Rate (%)", 5.0, 20.0, 10.0, 0.1, lambda x: f"{x:.1f}%", "Loan interest rate.")
            loan_income = slider_with_value("Loan Percent of Income", 0.01, 0.83, 0.20, 0.01, lambda x: f"{x:.0%}", "Loan amount as a fraction of annual income.")

        with right:
            credit_history = slider_with_value("Credit History Length (years)", 2, 30, 7, 1, lambda x: f"{x} years", "Length of credit history.")
            credit_score = slider_with_value("Credit Score", 390, 850, 680, 1, lambda x: f"{x}", "Credit score.")
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
                "person_age": int(age), "person_income": int(income), "person_home_ownership": home,
                "person_emp_exp": int(employment), "loan_intent": intent, "loan_amnt": int(loan_amount),
                "loan_int_rate": float(interest), "loan_percent_income": float(loan_income),
                "cb_person_cred_hist_length": int(credit_history), "credit_score": int(credit_score),
                "previous_loan_defaults_on_file": defaults, "person_gender": gender, "person_education": education,
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
                st.bar_chart(score_df, horizontal=True, height=250)
                st.markdown("#### Approval score")
                st.progress(float(result["approval_probability"]))

            with profile_col:
                st.markdown("#### Applicant snapshot")
                snapshot = pd.DataFrame({
                    "Field": ["Age", "Income", "Loan Amount", "Interest Rate", "Loan / Income", "Credit Score", "Credit History"],
                    "Selected value": [f"{age} years", f"${income:,.0f}", f"${loan_amount:,.0f}", f"{interest:.1f}%", f"{loan_income:.0%}", f"{credit_score}", f"{credit_history} years"],
                })
                st.dataframe(snapshot, hide_index=True, use_container_width=True)

            st.markdown("#### Model performance")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Accuracy", f'{metrics["accuracy"] * 100:.2f}%')
            p2.metric("Precision", f'{metrics["precision"] * 100:.2f}%')
            p3.metric("Recall", f'{metrics["recall"] * 100:.2f}%')
            p4.metric("F1 Score", f'{metrics["f1"] * 100:.2f}%')

            st.markdown("#### Where this model learned from")
            st.caption("These are the model's feature-importance values from XGBoost. They describe model contribution, not causation.")
            top_features = FEATURE_IMPORTANCE.head(8).sort_values("Importance")
            st.bar_chart(top_features, horizontal=True, height=300)

            st.caption("Displayed probabilities are model scores from XGBoost, not guaranteed real-world approval odds.")
        except Exception as e:
            st.error("The prediction could not be completed.")
            st.exception(e)

with tab2:
    st.subheader("CSV analysis dashboard")
    st.markdown('<div class="section-note">Upload applicant records, run the trained inference pipeline, explore the model-related charts, and download the predictions.</div>', unsafe_allow_html=True)
    st.write("")

    sample_path = Path("demo_test.csv")
    if sample_path.exists():
        st.download_button("Download Test CSV", data=sample_path.read_bytes(), file_name="loan_prediction_test.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
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

                st.success(f"Analysis complete: {len(results):,} applicants processed.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Applicants", f"{len(results):,}")
                c2.metric("Approved", f"{approved:,}")
                c3.metric("Rejected", f"{rejected:,}")
                c4.metric("Approval Rate", f"{approval_rate * 100:.1f}%")

                chart1, chart2 = st.columns(2, gap="large")
                with chart1:
                    st.markdown("#### Prediction distribution")
                    distribution = pd.DataFrame({"Applicants": [approved, rejected]}, index=["Approved", "Rejected"])
                    st.bar_chart(distribution, height=300)
                with chart2:
                    st.markdown("#### Approval score distribution")
                    st.line_chart(results["approval_probability"].reset_index(drop=True), height=300)

                st.markdown("#### Approval score bands")
                bands = pd.cut(results["approval_probability"], bins=[0, .25, .50, .75, 1.0], labels=["0–25%", "25–50%", "50–75%", "75–100%"], include_lowest=True)
                band_counts = bands.value_counts().sort_index()
                st.bar_chart(pd.DataFrame({"Applicants": band_counts}), height=280)

                if "loan_amnt" in df.columns:
                    scatter = pd.DataFrame({
                        "Loan Amount": pd.to_numeric(df["loan_amnt"], errors="coerce").reset_index(drop=True),
                        "Approval Score": results["approval_probability"].reset_index(drop=True),
                    }).dropna()
                    st.markdown("#### Loan amount vs approval score")
                    st.scatter_chart(scatter, x="Loan Amount", y="Approval Score", height=320)

                if "credit_score" in df.columns:
                    scatter_credit = pd.DataFrame({
                        "Credit Score": pd.to_numeric(df["credit_score"], errors="coerce").reset_index(drop=True),
                        "Approval Score": results["approval_probability"].reset_index(drop=True),
                    }).dropna()
                    st.markdown("#### Credit score vs approval score")
                    st.scatter_chart(scatter_credit, x="Credit Score", y="Approval Score", height=320)

                st.markdown("#### Prediction results")
                st.dataframe(results, use_container_width=True)
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button("Download Prediction Results", data=csv, file_name="loan_prediction_results.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error("Could not analyze this CSV. Make sure it contains the required model input columns.")
            st.exception(e)

with tab3:
    st.subheader("Model insights")
    st.markdown('<div class="section-note">Research-backed diagnostics from the completed model experiments. These charts describe the trained model and its held-out test performance.</div>', unsafe_allow_html=True)
    st.write("")

    a, b, c, d = st.columns(4)
    a.metric("Dataset", "45,000 rows")
    b.metric("Encoded Features", "19")
    c.metric("Models Benchmarked", "9")
    d.metric("Best Test Accuracy", "93.28%")

    st.divider()
    st.markdown("### 1. Model benchmark")
    st.caption("All nine models were evaluated on the same held-out test set.")
    benchmark_pct = BENCHMARK * 100
    st.bar_chart(benchmark_pct[["Accuracy"]], horizontal=True, height=430)

    st.markdown("#### Precision, Recall and F1 comparison")
    st.bar_chart(benchmark_pct[["Precision", "Recall", "F1 Score"]], height=430)

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### 2. XGBoost confusion matrix")
        st.caption("Held-out test set: 8,998 applicants.")
        st.dataframe(CONFUSION_MATRIX.style.background_gradient(axis=None), use_container_width=True)
        st.caption("TN = 6,753 | FP = 245 | FN = 360 | TP = 1,640")

    with right:
        st.markdown("### 3. Feature importance")
        st.caption("Model-native XGBoost importance; not a causal analysis.")
        st.bar_chart(FEATURE_IMPORTANCE.sort_values("Importance"), horizontal=True, height=430)

    st.divider()
    st.markdown("### 4. Class distribution")
    class_distribution = pd.DataFrame({"Applicants": [35000, 10000]}, index=["Rejected (0)", "Approved (1)"])
    st.bar_chart(class_distribution, height=280)
    st.caption("Original dataset distribution: 77.78% class 0 and 22.22% class 1. SMOTETomek was applied only to the training data.")

    st.markdown("### 5. XGBoost test metrics")
    xgb_metrics = pd.DataFrame({
        "Score": [0.9328, 0.8700, 0.8200, 0.8443]
    }, index=["Accuracy", "Precision", "Recall", "F1 Score"])
    st.bar_chart(xgb_metrics * 100, horizontal=True, height=260)

    with st.expander("XGBoost configuration"):
        st.code("""n_estimators = 900
max_depth = 5
learning_rate = 0.035
subsample = 0.95
colsample_bytree = 0.90
min_child_weight = 1
gamma = 0.0
reg_alpha = 0.02
reg_lambda = 1.0
eval_metric = logloss
tree_method = hist
random_state = 42""", language="text")

    st.caption("Feature importance does not mean that a feature alone causes approval or rejection. Model scores are not guaranteed real-world approval probabilities.")
