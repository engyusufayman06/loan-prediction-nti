import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.model import train_model, predict_one, predict_batch

st.set_page_config(page_title="Loan Intelligence", page_icon=None, layout="wide")

# -----------------------------------------------------------------------------
# Light UI
# -----------------------------------------------------------------------------
st.markdown("""
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


# -----------------------------------------------------------------------------
# Research results from the completed model experiments.
# -----------------------------------------------------------------------------
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


def chart_style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="x", alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9e2ea")
    ax.spines["bottom"].set_color("#d9e2ea")
    ax.tick_params(colors="#526273")


def horizontal_metric_chart(data, title, xlabel="Score (%)"):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    vals = data.iloc[:, 0].values
    names = data.index.tolist()
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color="#526273")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#18324a")
    ax.set_xlim(0, max(100, float(vals.max()) * 1.12))
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9, color="#526273")
    chart_style(ax)
    fig.tight_layout()
    return fig


tab1, tab2, tab3 = st.tabs(["Single Applicant", "CSV Analysis", "Model Insights"])

# -----------------------------------------------------------------------------
# Single applicant
# -----------------------------------------------------------------------------
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

            st.markdown("#### Global feature importance")
            st.caption("This is the trained XGBoost model's global feature importance. It is not a per-applicant explanation and does not imply causation.")
            st.pyplot(horizontal_metric_chart(FEATURE_IMPORTANCE.head(8).sort_values("Importance") * 100, "Top model features"), clear_figure=True)
            st.caption("Displayed probabilities are model scores from XGBoost, not guaranteed real-world approval odds.")
        except Exception as e:
            st.error("The prediction could not be completed.")
            st.exception(e)

# -----------------------------------------------------------------------------
# CSV analysis
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("CSV analysis dashboard")
    st.markdown('<div class="section-note">Upload applicant records, run the trained inference pipeline, explore model-related statistics, and download the predictions.</div>', unsafe_allow_html=True)
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

# -----------------------------------------------------------------------------
# Model insights
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Model insights")
    st.markdown('<div class="section-note">A deeper analytics dashboard built only from the completed model experiments and the held-out test results.</div>', unsafe_allow_html=True)
    st.write("")

    # Executive metrics
    best_model = BENCHMARK["Accuracy"].idxmax()
    best_accuracy = BENCHMARK.loc[best_model, "Accuracy"]
    runner_up = BENCHMARK["Accuracy"].sort_values(ascending=False).iloc[1]
    accuracy_gap = best_accuracy - runner_up
    a, b, c, d, e = st.columns(5)
    a.metric("Dataset", "45,000")
    b.metric("Encoded Features", "19")
    c.metric("Models", "9")
    d.metric("Best Accuracy", f"{best_accuracy * 100:.2f}%")
    e.metric("Best vs Runner-up", f"+{accuracy_gap * 100:.2f} pp")

    # Dataset / preprocessing statistics
    st.divider()
    st.markdown("### Dataset and training pipeline")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Original rows", "45,000")
    p2.metric("Training rows", "35,992")
    p3.metric("After SMOTETomek", "55,798")
    p4.metric("Held-out test rows", "8,998")

    pipeline_counts = pd.DataFrame({"Rows": [45000, 35992, 55798, 8998]}, index=["Original dataset", "Train split", "Balanced train", "Test split"])
    st.bar_chart(pipeline_counts, height=300)
    st.caption("The balanced training count is larger than the original training count because SMOTETomek was applied only to the training data. The test set remains held out.")

    # Class balance
    class_col, class_stats = st.columns([1.35, 1], gap="large")
    with class_col:
        st.markdown("### Class balance")
        class_distribution = pd.Series([35000, 10000], index=["Rejected (0)", "Approved (1)"], name="Applicants")
        st.bar_chart(class_distribution, height=300)
    with class_stats:
        st.markdown("### Class statistics")
        st.metric("Rejected share", "77.78%")
        st.metric("Approved share", "22.22%")
        st.metric("Class ratio", "3.50 : 1")
        st.caption("This imbalance is one reason the workflow evaluates precision, recall and F1 rather than accuracy alone.")

    # Benchmark dashboard
    st.divider()
    st.markdown("### Model benchmark leaderboard")
    st.caption("All nine models were evaluated on the same held-out test set.")
    ranked = BENCHMARK.sort_values("Accuracy", ascending=False)
    st.pyplot(horizontal_metric_chart(ranked[["Accuracy"]] * 100, "Test accuracy by model"), clear_figure=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Precision / Recall / F1")
        st.bar_chart((ranked[["Precision", "Recall", "F1 Score"]] * 100), height=430)
    with right:
        st.markdown("#### Full benchmark table")
        display_benchmark = (ranked * 100).round(2)
        st.dataframe(display_benchmark, use_container_width=True)

    # Metric heatmap
    st.markdown("#### Benchmark metric heatmap")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    heat = ranked[["Accuracy", "Precision", "Recall", "F1 Score"]].values * 100
    im = ax.imshow(heat, aspect="auto")
    ax.set_xticks(range(4), ["Accuracy", "Precision", "Recall", "F1 Score"])
    ax.set_yticks(range(len(ranked)), ranked.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Model performance matrix", loc="left", fontsize=14, fontweight="bold", color="#18324a")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Score (%)")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Precision-recall tradeoff
    st.markdown("#### Precision vs Recall trade-off")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(BENCHMARK["Recall"] * 100, BENCHMARK["Precision"] * 100, s=80)
    for name, row in BENCHMARK.iterrows():
        ax.annotate(name, (row["Recall"] * 100, row["Precision"] * 100), xytext=(6, 5), textcoords="offset points", fontsize=8, color="#526273")
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision–Recall trade-off across models", loc="left", fontsize=14, fontweight="bold", color="#18324a")
    chart_style(ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # XGBoost diagnostics
    st.divider()
    st.markdown("### XGBoost diagnostic dashboard")
    xgb_acc = 0.9328
    xgb_precision = 0.8700
    xgb_recall = 0.8200
    xgb_f1 = 0.8443
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "93.28%")
    m2.metric("Precision", "87.00%")
    m3.metric("Recall", "82.00%")
    m4.metric("F1 Score", "84.43%")

    xgb_metrics = pd.Series([xgb_acc, xgb_precision, xgb_recall, xgb_f1], index=["Accuracy", "Precision", "Recall", "F1 Score"], name="Score")
    st.bar_chart(xgb_metrics * 100, height=280)

    # Confusion matrix and error rates
    cm_left, cm_right = st.columns(2, gap="large")
    with cm_left:
        st.markdown("#### Confusion matrix")
        st.caption("Held-out test set: 8,998 applicants.")
        st.dataframe(CONFUSION_MATRIX.style.background_gradient(axis=None), use_container_width=True)
        st.caption("TN = 6,753 | FP = 245 | FN = 360 | TP = 1,640")
    with cm_right:
        st.markdown("#### Error analysis")
        tn, fp, fn, tp = 6753, 245, 360, 1640
        error_stats = pd.Series({
            "False positive rate": fp / (fp + tn),
            "False negative rate": fn / (fn + tp),
            "Overall error rate": (fp + fn) / (tn + fp + fn + tp),
            "Correct prediction rate": (tn + tp) / (tn + fp + fn + tp),
        })
        st.bar_chart(error_stats * 100, height=280)
        st.caption("Rates are calculated directly from the reported confusion matrix.")

    # Per-class metrics
    st.markdown("#### Per-class performance")
    per_class = pd.DataFrame({
        "Precision": [0.95, 0.87],
        "Recall": [0.96, 0.82],
        "F1 Score": [0.96, 0.84],
        "Support": [6998, 2000],
    }, index=["Rejected (0)", "Approved (1)"])
    st.dataframe(per_class, use_container_width=True)
    st.bar_chart(per_class[["Precision", "Recall", "F1 Score"]] * 100, height=300)

    # Feature importance
    st.divider()
    st.markdown("### What does XGBoost rely on?")
    st.caption("Model-native feature importance. Importance is not causal and does not mean a feature independently determines approval.")
    feature_pct = FEATURE_IMPORTANCE["Importance"] * 100
    st.pyplot(horizontal_metric_chart(FEATURE_IMPORTANCE.sort_values("Importance") * 100, "All XGBoost feature importance", "Importance (%)"), clear_figure=True)

    top5 = FEATURE_IMPORTANCE.head(5)["Importance"].sum()
    top1 = FEATURE_IMPORTANCE.iloc[0]["Importance"]
    f1, f2, f3 = st.columns(3)
    f1.metric("Top feature", "Previous loan defaults")
    f2.metric("Top feature importance", f"{top1 * 100:.2f}%")
    f3.metric("Top 5 cumulative importance", f"{top5 * 100:.2f}%")

    # Cumulative importance
    cumulative = FEATURE_IMPORTANCE["Importance"].sort_values(ascending=False).cumsum() * 100
    cumulative.index = [f"Top {i}" for i in range(1, len(cumulative) + 1)]
    st.markdown("#### Cumulative feature importance")
    st.line_chart(cumulative, height=300)
    st.caption("This cumulative view is a mathematical summary of the reported XGBoost feature-importance values.")

    # Feature groups
    group_values = pd.Series({
        "Previous defaults": 0.805732,
        "Home ownership": 0.031212 + 0.020834,
        "Loan / credit factors": 0.018895 + 0.016857 + 0.012999,
        "Education": 0.020783,
        "Loan intent": 0.017310 + 0.008511 + 0.007039 + 0.005432 + 0.004117,
        "Personal / employment": 0.009649 + 0.004760 + 0.004585,
    }).sort_values(ascending=False) * 100
    st.markdown("#### Feature importance by broad group")
    st.bar_chart(group_values, height=320)
    st.caption("Groups are simple sums of the displayed feature-importance values and are intended only as a visualization aid.")

    # Confusion matrix visual
    st.markdown("#### Confusion matrix visual")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    cm = CONFUSION_MATRIX.values
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks([0, 1], ["Predicted Rejected", "Predicted Approved"])
    ax.set_yticks([0, 1], ["Actual Rejected", "Actual Approved"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("XGBoost test confusion matrix", loc="left", fontsize=14, fontweight="bold", color="#18324a")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Applicants")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Model configuration
    st.divider()
    st.markdown("### XGBoost configuration")
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

    # Research notes
    st.markdown("### Research notes")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown("**Generalization**")
        st.write("The reported benchmark uses a held-out test set. Training accuracy is not reported here, so no unsupported train/test overfitting claim is made.")
    with n2:
        st.markdown("**Imbalance handling**")
        st.write("SMOTETomek is applied to training data only. The held-out test set remains separate for evaluation.")
    with n3:
        st.markdown("**Interpretation**")
        st.write("Feature importance describes model behavior, not causal relationships or guaranteed real-world approval odds.")

    st.caption("All research statistics shown in this tab come from the completed model experiments. No ROC or calibration chart is fabricated because the underlying notebook does not provide the required prediction arrays for those curves.")
