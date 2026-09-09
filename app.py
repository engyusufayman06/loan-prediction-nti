from pathlib import Path
import html

import pandas as pd
import streamlit as st

from src.model import MODEL_INPUT_COLUMNS, predict_batch, predict_one, train_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_data.csv"

st.set_page_config(
    page_title="Loan Intelligence | NTI ML",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --bg:#03060b; --panel:#080e17; --line:#1a2b3f; --text:#f7fafc; --muted:#8796a9; --cyan:#22d3ee; --lime:#a3e635; --red:#fb7185; }
    .stApp { background:var(--bg); color:var(--text); }
    [data-testid="stSidebar"] { background:#050a11; border-right:1px solid var(--line); }
    .block-container { max-width:1450px; padding:1.5rem 2rem 4rem; }
    .hero { border:1px solid var(--line); border-radius:24px; padding:2.2rem; background:radial-gradient(circle at 90% 10%,rgba(34,211,238,.16),transparent 30%),linear-gradient(135deg,#07111c,#0b1020); margin-bottom:1.2rem; }
    .eyebrow { color:var(--cyan); font-size:.65rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
    .hero h1 { margin:.4rem 0; font-size:clamp(2.8rem,6vw,5.4rem); line-height:.9; letter-spacing:-.07em; }
    .hero h1 span { color:var(--cyan); }
    .hero p { color:#aab8c8; max-width:850px; line-height:1.7; }
    .card { border:1px solid var(--line); border-radius:16px; padding:1rem; background:linear-gradient(145deg,#09111b,#070c14); height:100%; }
    .card-k { color:#718198; font-size:.58rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
    .card-v { font-family:monospace; font-size:1.6rem; margin-top:.25rem; }
    .card-note { color:var(--muted); font-size:.65rem; margin-top:.25rem; }
    .result { border:1px solid var(--line); border-radius:20px; padding:2rem; text-align:center; background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.12),transparent 45%),var(--panel); }
    .approved { color:var(--lime); font-size:2.8rem; font-weight:800; }
    .rejected { color:var(--red); font-size:2.8rem; font-weight:800; }
    .muted { color:var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("loan_data.csv was not found in the project root.")
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training XGBoost model…")
def load_model():
    return train_model(DATA_PATH)


def metric_card(title, value, note):
    return f'<div class="card"><div class="card-k">{html.escape(str(title))}</div><div class="card-v">{html.escape(str(value))}</div><div class="card-note">{html.escape(str(note))}</div></div>'


def render_overview(reference, metrics):
    st.markdown(
        '<div class="hero"><div class="eyebrow">NTI Machine Learning Track · End-to-End Classification</div>'
        '<h1>LOAN <span>INTELLIGENCE</span></h1>'
        '<p>Applicant data → preprocessing → class-imbalance handling → XGBoost → explainable prediction output. '
        'This is an educational ML prototype, not a production underwriting system.</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Dataset rows", f"{len(reference):,}", "source records"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Model inputs", len(MODEL_INPUT_COLUMNS), "raw input columns"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%", "held-out test set"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Balanced train", f"{metrics['train_rows_after_smotetomek']:,}", "after SMOTETomek"), unsafe_allow_html=True)

    st.subheader("ML Pipeline")
    st.code(
        "Raw data → remove loan_id → IQR clipping → encoding → 80/20 split → "
        "RobustScaler → SMOTETomek (train only) → XGBoost → prediction",
        language="text",
    )

    st.subheader("Held-out performance")
    report = metrics["classification_report"]
    rows = []
    for name in ["0", "1"]:
        rows.append({
            "Class": name,
            "Precision": report[name]["precision"],
            "Recall": report[name]["recall"],
            "F1": report[name]["f1-score"],
            "Support": int(report[name]["support"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_prediction(reference, bundle):
    st.title("Prediction Studio")
    st.caption("Enter one applicant record and run the trained XGBoost model.")

    home_values = sorted(reference["person_home_ownership"].dropna().astype(str).unique())
    intent_values = sorted(reference["loan_intent"].dropna().astype(str).unique())
    gender_values = sorted(reference["person_gender"].dropna().astype(str).unique())
    default_values = sorted(reference["previous_loan_defaults_on_file"].dropna().astype(str).unique())
    education_values = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]

    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
        income = st.number_input("Annual income", min_value=0.0, value=50000.0, step=1000.0)
        emp_exp = st.number_input("Employment experience (years)", min_value=0, max_value=60, value=5, step=1)
        home = st.selectbox("Home ownership", home_values)
        intent = st.selectbox("Loan intent", intent_values)
        loan_amount = st.number_input("Loan amount", min_value=0.0, value=10000.0, step=500.0)
        interest = st.number_input("Interest rate (%)", min_value=0.0, value=10.0, step=0.1)
    with right:
        loan_percent_income = st.number_input("Loan percent of income", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        credit_history = st.number_input("Credit history length", min_value=0.0, value=5.0, step=0.5)
        credit_score = st.number_input("Credit score", min_value=0, max_value=1000, value=650, step=1)
        previous_default = st.selectbox("Previous loan defaults", default_values)
        gender = st.selectbox("Gender", gender_values)
        education = st.selectbox("Education", education_values)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        applicant = {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home,
            "person_emp_exp": emp_exp,
            "loan_intent": intent,
            "loan_amnt": loan_amount,
            "loan_int_rate": interest,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": credit_history,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_default,
            "person_gender": gender,
            "person_education": education,
        }
        result = predict_one(bundle, applicant)
        label_class = "approved" if result["prediction"] == 1 else "rejected"
        st.markdown(
            f'<div class="result"><div class="muted">MODEL DECISION</div>'
            f'<div class="{label_class}">{html.escape(result["label"])}</div>'
            f'<p>Approval probability: <b>{result["approval_probability"]:.2%}</b> · '
            f'Rejection probability: <b>{result["rejection_probability"]:.2%}</b></p></div>',
            unsafe_allow_html=True,
        )


def render_batch(bundle):
    st.title("Batch Lab")
    st.caption("Upload a CSV containing the required model input columns.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        st.info("Required columns: " + ", ".join(MODEL_INPUT_COLUMNS))
        return

    data = pd.read_csv(uploaded)
    st.write(f"Loaded **{len(data):,}** rows.")
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    st.dataframe(data.head(10), use_container_width=True)
    if st.button("Run Batch Prediction", type="primary"):
        with st.spinner("Running predictions…"):
            result = predict_batch(bundle, data)
        st.success(f"Completed {len(result):,} predictions.")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions CSV",
            result.to_csv(index=False).encode("utf-8"),
            "loan_predictions.csv",
            "text/csv",
        )


def render_insights(reference, metrics):
    st.title("Insights")
    st.caption("Quick facts from the dataset and evaluation run.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Target distribution")
        st.bar_chart(reference["loan_status"].value_counts().sort_index())
    with c2:
        st.subheader("Confusion matrix")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm, use_container_width=True)

    st.subheader("Numeric summary")
    numeric = reference.select_dtypes(include="number")
    st.dataframe(numeric.describe().T, use_container_width=True)


def render_presentation():
    st.title("Project Presentation")
    st.caption("A compact judge-ready story for the Loan Intelligence project.")
    slides = [
        ("01 · Problem", "Loan status is a binary classification task: transform applicant and loan signals into a repeatable prediction."),
        ("02 · Data", "The dataset contains applicant, employment, loan, credit and target information. loan_id is treated as an identifier, not a predictive feature."),
        ("03 · Preprocessing", "IQR clipping, age/employment sanity filters, binary and ordinal mappings, one-hot encoding, then RobustScaler."),
        ("04 · Imbalance", "SMOTETomek is applied only to the training data so the held-out test set remains a fair evaluation set."),
        ("05 · Benchmark", "Six classifiers were compared. XGBoost achieved the strongest held-out accuracy and class-1 F1 in the project benchmark."),
        ("06 · Result", "The documented XGBoost benchmark accuracy is 92.87%. The model is a portfolio/educational prototype, not a real lending decision engine."),
        ("07 · Next", "Production hardening would add calibration, fairness analysis, SHAP explanations, model versioning, monitoring, and a reproducible deployment pipeline."),
    ]
    for title, body in slides:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.write(body)


def render_team():
    st.title("Team")
    names = [
        "Yusuf Ayman Tolba",
        "Mohamed Reda Hussein",
        "Abdelrahman Mohamed Ahmed",
        "Abdelmoniem Ibrahim Abdelmoniem",
        "Asmaa Rabee Mohammed",
    ]
    for i, name in enumerate(names, 1):
        st.markdown(f"**{i:02d} · {name}**")


try:
    reference = load_data()
    bundle, metrics = load_model()
except Exception as exc:
    st.error(f"Project startup failed: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## ◈ Loan Intelligence")
    st.caption("NTI Machine Learning Track")
    page = st.radio(
        "Navigation",
        ["Overview", "Prediction Studio", "Batch Lab", "Insights", "Project Presentation", "Team"],
    )
    st.divider()
    st.caption("Educational ML prototype · Not real underwriting")

if page == "Overview":
    render_overview(reference, metrics)
elif page == "Prediction Studio":
    render_prediction(reference, bundle)
elif page == "Batch Lab":
    render_batch(bundle)
elif page == "Insights":
    render_insights(reference, metrics)
elif page == "Project Presentation":
    render_presentation()
elif page == "Team":
    render_team()
