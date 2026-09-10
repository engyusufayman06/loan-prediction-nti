from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.model import ENGINEERED_FEATURES, predict_batch, predict_one, train_model


st.set_page_config(
    page_title="Loan Intelligence",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
<style>
    .stApp { background: #ffffff; color: #18324a; }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 4rem; }
    .main-title { font-size: 2.55rem; font-weight: 750; color: #18324a; letter-spacing: -0.03em; margin-bottom: .15rem; }
    .sub-title { color: #6b7b8c; font-size: 1rem; margin-bottom: 1.8rem; }
    .section-note { background: #f5f8fb; border: 1px solid #e1e8ef; border-radius: 12px; padding: .8rem 1rem; color: #526273; }
    .value-pill { display: inline-block; background: #eef5ff; border: 1px solid #d7e6fb; color: #245a9a; border-radius: 8px; padding: 3px 9px; font-size: .82rem; font-weight: 650; margin-top: -4px; margin-bottom: 8px; }
    div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e2e8ef; border-radius: 13px; padding: .75rem 1rem; }
    div[data-baseweb="select"] > div { background: #ffffff; border-color: #d5dee7; }
    .stButton > button, .stFormSubmitButton > button { background: #2f6f8f; color: #ffffff; border: 0; border-radius: 9px; font-weight: 650; min-height: 2.7rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background: #255b75; color: #ffffff; }
    .stDownloadButton > button { border-radius: 9px; border: 1px solid #cfd9e2; background: #ffffff; color: #29465a; }
    hr { border-color: #e6ebf0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Loan Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">AI-powered credit decision and model analytics</div>',
    unsafe_allow_html=True,
)

DATA_PATH = Path("loan_data.csv")


@st.cache_resource(show_spinner="Preparing the model for the first prediction...")
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("loan_data.csv was not found in the project folder.")
    return train_model(DATA_PATH)


def slider_with_value(label, min_value, max_value, value, step, formatter, help_text):
    selected = st.slider(label, min_value, max_value, value, step, help=help_text)
    st.markdown(
        f'<div class="value-pill">Selected: {formatter(selected)}</div>',
        unsafe_allow_html=True,
    )
    return selected


# Final notebook benchmark — same 80/20 stratified test split.
BENCHMARK = pd.DataFrame(
    [
        ["HistGradientBoosting", 0.932874, 0.860537, 0.8330, 0.846545, 0.976527],
        ["XGBoost", 0.929651, 0.855434, 0.8225, 0.838644, 0.975957],
        ["Random Forest", 0.911202, 0.773079, 0.8500, 0.809717, 0.968021],
        ["Extra Trees", 0.897755, 0.738305, 0.8365, 0.784341, 0.963371],
        ["Gradient Boosting", 0.893532, 0.717809, 0.8585, 0.781876, 0.962550],
        ["SVM", 0.877639, 0.671630, 0.8795, 0.761637, 0.950739],
        ["Decision Tree", 0.880862, 0.703152, 0.8030, 0.749767, 0.853058],
        ["Logistic Regression", 0.861747, 0.635971, 0.8840, 0.739749, 0.944234],
        ["KNN", 0.862192, 0.643396, 0.8525, 0.733333, 0.926571],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
).set_index("Model")

NOTEBOOK_TOP_FEATURES = pd.Series(
    {
        "previous_loan_defaults_on_file": 88.0,
        "high_risk_income_ratio": 1.9,
        "cb_person_cred_hist_length": 1.4,
        "loan_int_rate": 1.0,
        "person_home_ownership": 0.8,
        "person_education": 0.7,
        "loan_intent": 0.7,
        "person_gender": 0.6,
    },
    name="Relative Importance (%)",
)

XGB_CONFUSION_MATRIX = pd.DataFrame(
    [[6720, 278], [355, 1645]],
    index=["Actual Rejected (0)", "Actual Approved (1)"],
    columns=["Predicted Rejected (0)", "Predicted Approved (1)"],
)

NOTEBOOK_DATASET_STATS = {
    "raw_rows": 45000,
    "raw_columns": 14,
    "clean_rows": 44988,
    "model_features": 18,
    "train_rows": 35990,
    "test_rows": 8998,
    "smote_rows": 55980,
    "class_0": 35000,
    "class_1": 10000,
    "train_class_0": 27990,
    "train_class_1": 8000,
}

OPTIMIZED_XGB = {
    "n_estimators": 250,
    "max_depth": 12,
    "learning_rate": 0.01,
    "subsample": 0.60,
    "colsample_bytree": 0.80,
    "gamma": 0.0,
    "reported_accuracy": 0.91,
    "reported_precision": 0.77,
    "reported_recall": 0.86,
    "reported_f1": 0.81,
}


def horizontal_chart(series, title, xlabel="Relative Importance (%)"):
    data = series.sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(data))
    ax.barh(y, data.values, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(data.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    for i, value in enumerate(data.values):
        ax.text(value + max(data.values) * 0.01, i, f"{value:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


tabs = st.tabs(["Single Applicant", "CSV Analysis", "Model Insights"])

with tabs[0]:
    st.subheader("Applicant profile")
    st.markdown(
        '<div class="section-note">Enter the 13 raw applicant fields used by the final notebook. The five engineered features are generated automatically by the shared ML core.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("loan_form"):
        left, right = st.columns(2, gap="large")
        with left:
            age = slider_with_value("Age", 18, 80, 28, 1, lambda x: f"{x} years", "Applicant age.")
            income = slider_with_value("Annual Income ($)", 8000, 720000, 60000, 1000, lambda x: f"${x:,.0f}", "Annual income.")
            employment = slider_with_value("Employment Experience (years)", 0, 50, 6, 1, lambda x: f"{x} years", "Employment experience.")
            loan_amount = slider_with_value("Loan Amount ($)", 500, 35000, 12000, 100, lambda x: f"${x:,.0f}", "Requested loan amount.")
            interest = slider_with_value("Interest Rate (%)", 5.0, 20.0, 10.0, 0.1, lambda x: f"{x:.1f}%", "Loan interest rate.")
            loan_income = slider_with_value("Loan Percent of Income", 0.00, 0.66, 0.20, 0.01, lambda x: f"{x:.0%}", "Loan amount as a fraction of annual income.")
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
                score_df = pd.DataFrame(
                    {"Score": [result["approval_probability"], result["rejection_probability"]]},
                    index=["Approval", "Rejection"],
                )
                st.bar_chart(score_df, horizontal=True, height=250)
            with profile_col:
                st.markdown("#### Applicant snapshot")
                snapshot = pd.DataFrame(
                    {
                        "Field": ["Age", "Income", "Loan Amount", "Interest Rate", "Loan / Income", "Credit Score", "Credit History"],
                        "Selected value": [
                            f"{age} years", f"${income:,.0f}", f"${loan_amount:,.0f}",
                            f"{interest:.1f}%", f"{loan_income:.0%}", f"{credit_score}", f"{credit_history} years",
                        ],
                    }
                )
                st.dataframe(snapshot, hide_index=True, use_container_width=True)

            st.markdown("#### Runtime model metrics")
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Accuracy", f'{metrics["accuracy"] * 100:.2f}%')
            p2.metric("Precision", f'{metrics["precision"] * 100:.2f}%')
            p3.metric("Recall", f'{metrics["recall"] * 100:.2f}%')
            p4.metric("F1 Score", f'{metrics["f1"] * 100:.2f}%')
            p5.metric("ROC-AUC", f'{metrics["roc_auc"]:.4f}')

            st.markdown("#### Engineered features")
            st.dataframe(pd.DataFrame({"Feature": ENGINEERED_FEATURES}), hide_index=True, use_container_width=True)
            st.caption("Probabilities are model scores, not calibrated real-world approval odds.")
        except Exception as exc:
            st.error("The prediction could not be completed.")
            st.exception(exc)

with tabs[1]:
    st.subheader("CSV analysis dashboard")
    st.markdown(
        '<div class="section-note">Upload applicant records using the same 13-field raw schema. Engineered features are created automatically, then the trained model returns predictions and probabilities.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

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
                    st.bar_chart(pd.DataFrame({"Applicants": [approved, rejected]}, index=["Approved", "Rejected"]), height=300)
                with chart2:
                    st.markdown("#### Approval score distribution")
                    st.line_chart(results["approval_probability"].reset_index(drop=True), height=300)

                st.markdown("#### Approval score bands")
                bands = pd.cut(
                    results["approval_probability"],
                    bins=[0, .25, .50, .75, 1.0],
                    labels=["0–25%", "25–50%", "50–75%", "75–100%"],
                    include_lowest=True,
                )
                st.bar_chart(pd.DataFrame({"Applicants": bands.value_counts().sort_index()}), height=280)

                st.markdown("#### Scored rows")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="loan_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error("The CSV could not be analyzed.")
            st.exception(exc)

with tabs[2]:
    st.subheader("Model Insights")
    st.markdown(
        '<div class="section-note">Research metrics and diagnostics from <code>final-loan-2.ipynb</code>. The benchmark uses the same stratified held-out test set across the nine classifiers.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset and pipeline")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Raw rows", f'{NOTEBOOK_DATASET_STATS["raw_rows"]:,}')
    d2.metric("Clean rows", f'{NOTEBOOK_DATASET_STATS["clean_rows"]:,}')
    d3.metric("Train rows", f'{NOTEBOOK_DATASET_STATS["train_rows"]:,}')
    d4.metric("Test rows", f'{NOTEBOOK_DATASET_STATS["test_rows"]:,}')
    d5.metric("Model features", f'{NOTEBOOK_DATASET_STATS["model_features"]}')

    d6, d7, d8, d9 = st.columns(4)
    d6.metric("SMOTE train rows", f'{NOTEBOOK_DATASET_STATS["smote_rows"]:,}')
    d7.metric("Original class 0", f'{NOTEBOOK_DATASET_STATS["class_0"]:,}')
    d8.metric("Original class 1", f'{NOTEBOOK_DATASET_STATS["class_1"]:,}')
    d9.metric("Class-1 share", f'{NOTEBOOK_DATASET_STATS["class_1"] / NOTEBOOK_DATASET_STATS["raw_rows"] * 100:.2f}%')

    st.caption("Final notebook cleaning keeps age ≤ 80 and employment experience ≤ 50, then adds five engineered features and applies LabelEncoder to object columns.")

    st.markdown("### Nine-model benchmark")
    display_benchmark = BENCHMARK.copy()
    display_benchmark["Accuracy"] = display_benchmark["Accuracy"].map(lambda x: f"{x * 100:.2f}%")
    display_benchmark["Precision"] = display_benchmark["Precision"].map(lambda x: f"{x:.4f}")
    display_benchmark["Recall"] = display_benchmark["Recall"].map(lambda x: f"{x:.4f}")
    display_benchmark["F1 Score"] = display_benchmark["F1 Score"].map(lambda x: f"{x:.4f}")
    display_benchmark["ROC-AUC"] = display_benchmark["ROC-AUC"].map(lambda x: f"{x:.6f}")
    st.dataframe(display_benchmark, use_container_width=True)

    chart_col, metric_col = st.columns(2, gap="large")
    with chart_col:
        st.markdown("#### Accuracy leaderboard")
        st.bar_chart(BENCHMARK["Accuracy"].sort_values(ascending=True) * 100, horizontal=True, height=360)
    with metric_col:
        st.markdown("#### F1 leaderboard")
        st.bar_chart(BENCHMARK["F1 Score"].sort_values(ascending=True) * 100, horizontal=True, height=360)

    st.markdown("### XGBoost diagnostics")
    x1, x2, x3, x4, x5 = st.columns(5)
    xgb_row = BENCHMARK.loc["XGBoost"]
    x1.metric("Accuracy", f'{xgb_row["Accuracy"] * 100:.2f}%')
    x2.metric("Precision", f'{xgb_row["Precision"]:.4f}')
    x3.metric("Recall", f'{xgb_row["Recall"]:.4f}')
    x4.metric("F1", f'{xgb_row["F1 Score"]:.4f}')
    x5.metric("ROC-AUC", f'{xgb_row["ROC-AUC"]:.6f}')

    cm1, cm2 = st.columns(2, gap="large")
    with cm1:
        st.markdown("#### Confusion matrix")
        st.dataframe(XGB_CONFUSION_MATRIX, use_container_width=True)
    with cm2:
        tn, fp, fn, tp = 6720, 278, 355, 1645
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        error_rate = (fp + fn) / (tn + fp + fn + tp)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("False Positive Rate", f"{fpr * 100:.2f}%")
        q2.metric("False Negative Rate", f"{fnr * 100:.2f}%")
        q3.metric("Error Rate", f"{error_rate * 100:.2f}%")
        q4.metric("Correct Rate", f"{(1 - error_rate) * 100:.2f}%")

    st.markdown("### XGBoost top features")
    st.caption("Relative importance from the notebook's global XGBoost feature-importance chart. These values describe model contribution, not causation.")
    st.pyplot(horizontal_chart(NOTEBOOK_TOP_FEATURES, "Top XGBoost features"), clear_figure=True)
    st.dataframe(
        NOTEBOOK_TOP_FEATURES.rename("Importance (%)").to_frame().style.format("{:.1f}%"),
        use_container_width=True,
    )

    st.markdown("### Hyperparameter optimization experiment")
    st.caption("The notebook also runs a separate RandomizedSearchCV experiment scored by F1. Its reported test classification report is shown here as a secondary experiment, not as the deployed benchmark model.")
    params_df = pd.DataFrame(
        {
            "Parameter": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "gamma"],
            "Optimal value": [250, 12, 0.01, 0.60, 0.80, 0.0],
        }
    )
    st.dataframe(params_df, hide_index=True, use_container_width=True)
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Reported Accuracy", "91%")
    o2.metric("Reported Precision", "0.77")
    o3.metric("Reported Recall", "0.86")
    o4.metric("Reported F1", "0.81")

    st.markdown("### Important interpretation notes")
    st.write(
        "The benchmark leader in the final notebook is HistGradientBoosting (93.2874% accuracy and 0.846545 F1). "
        "XGBoost is the project's deployable model and reaches 92.9651% accuracy, 0.855434 precision, "
        "0.8225 recall, 0.838644 F1, and 0.975957 ROC-AUC on the same test split."
    )
    st.write(
        "The five engineered features are income_to_loan_ratio, emp_length_to_age_ratio, "
        "cred_hist_to_age_ratio, high_risk_income_ratio, and is_renting."
    )
