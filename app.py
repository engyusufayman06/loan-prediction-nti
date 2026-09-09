from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from src.model import predict_one, train_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_data.csv"

st.set_page_config(
    page_title="Loan Intelligence | NTI ML",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root{--bg:#050810;--panel:#0a101a;--panel2:#0d1521;--line:#1b2a3c;--text:#f4f7fb;--muted:#8493a8;--cyan:#22d3ee;--lime:#a3e635;--purple:#a78bfa;--danger:#fb7185;--amber:#f59e0b}
    html,body,[class*="css"]{font-family:'Manrope',sans-serif}
    .stApp{background:var(--bg);color:var(--text)}
    [data-testid="stHeader"]{background:rgba(5,8,16,.88)}
    [data-testid="stSidebar"]{background:#070c14;border-right:1px solid var(--line)}
    .block-container{max-width:1450px;padding:1.5rem 2.25rem 3.5rem}
    #MainMenu,footer{visibility:hidden}
    .mono{font-family:'DM Mono',monospace;letter-spacing:.08em}
    .eyebrow{color:var(--cyan);font-size:.68rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase}
    .brand{padding:.3rem .15rem 1.2rem}.brand-mark{color:var(--cyan);font-size:1.25rem;font-weight:800}.brand-title{font-size:1rem;font-weight:800;margin-top:.2rem}.brand-sub{color:var(--muted);font-size:.7rem;margin-top:.2rem}
    .nav-label{font-size:.64rem;color:#627187;text-transform:uppercase;letter-spacing:.16em;font-weight:800;margin:1.1rem 0 .4rem}
    .hero{position:relative;overflow:hidden;min-height:235px;padding:2.25rem 2.4rem;border:1px solid var(--line);border-radius:26px;background:radial-gradient(circle at 87% 15%,rgba(34,211,238,.14),transparent 25%),radial-gradient(circle at 70% 110%,rgba(167,139,250,.12),transparent 31%),linear-gradient(135deg,#09111d,#08101a 60%,#0c1020);margin-bottom:1.25rem}
    .hero:after{content:'';position:absolute;right:-85px;top:-120px;width:320px;height:320px;border:1px solid rgba(34,211,238,.15);border-radius:50%;box-shadow:0 0 0 30px rgba(34,211,238,.025),0 0 0 62px rgba(34,211,238,.014)}
    .hero h1{margin:.35rem 0 0;font-size:clamp(2.4rem,5vw,4.7rem);line-height:.92;letter-spacing:-.07em;font-weight:800}.hero h1 span{color:var(--cyan)}
    .hero p{color:#9aa8ba;max-width:760px;margin:.95rem 0 0;font-size:.92rem;line-height:1.7}
    .hero-meta{margin-top:1.35rem;display:flex;gap:.55rem;flex-wrap:wrap}.pill{border:1px solid var(--line);background:rgba(255,255,255,.025);color:#aeb9c8;padding:.4rem .65rem;border-radius:999px;font-size:.67rem}.pill strong{color:var(--text)}
    .section-title{margin:1.15rem 0 .65rem;font-size:.68rem;text-transform:uppercase;letter-spacing:.16em;color:#8fa0b5;font-weight:800}
    .page-title{font-size:2.15rem;font-weight:800;letter-spacing:-.05em;margin:.15rem 0 .25rem}.page-sub{color:var(--muted);font-size:.82rem;margin-bottom:1.2rem}
    .card{border:1px solid var(--line);background:linear-gradient(145deg,#0a111b,#090f18);border-radius:18px;padding:1.15rem 1.2rem;height:100%}
    .card-k{color:#718198;font-size:.62rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800}.card-v{font-family:'DM Mono',monospace;font-size:1.55rem;margin-top:.3rem}.card-note{color:var(--muted);font-size:.68rem;margin-top:.3rem;line-height:1.5}
    .step{display:flex;gap:.85rem;align-items:flex-start;padding:.82rem 0;border-bottom:1px solid #132031}.step:last-child{border-bottom:0}.step-no{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.72rem;border:1px solid #214051;border-radius:8px;padding:.32rem .42rem}.step h4{margin:0;color:#e8edf4;font-size:.78rem}.step p{margin:.18rem 0 0;color:var(--muted);font-size:.68rem;line-height:1.5}
    label,.stNumberInput label,.stSelectbox label{color:#b7c1ce!important;font-size:.7rem!important;font-weight:600!important}
    input,[data-baseweb="select"]>div{background:#080e18!important;border-color:#1b2b3e!important;color:var(--text)!important;border-radius:10px!important}
    input:focus{border-color:rgba(34,211,238,.65)!important;box-shadow:0 0 0 1px rgba(34,211,238,.15)!important}
    .stButton>button{border:1px solid rgba(34,211,238,.35);background:linear-gradient(135deg,#0d202a,#0a151e);color:#dffbff;border-radius:11px;min-height:45px;font-weight:800}.stButton>button:hover{border-color:var(--cyan);color:white}
    .decision{min-height:280px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border:1px solid var(--line);border-radius:20px;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.09),transparent 48%),var(--panel);padding:1.5rem}.decision .status{font-size:.64rem;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:800}.decision .label{margin-top:.45rem;font-size:2.55rem;line-height:1;font-weight:800;letter-spacing:-.055em}.approved{color:var(--lime)}.rejected{color:var(--danger)}.prob{font-family:'DM Mono',monospace;margin-top:.8rem;font-size:1.08rem;color:#d6e3ef}.confidence-bar{width:82%;height:6px;background:#172232;border-radius:99px;overflow:hidden;margin-top:1rem}.confidence-fill{height:100%;background:var(--cyan);border-radius:99px}.decision-note{color:var(--muted);font-size:.65rem;margin-top:.75rem}
    .metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem;margin-top:.75rem}.metric{border:1px solid var(--line);background:#090f19;border-radius:13px;padding:.82rem .9rem}.metric .k{color:var(--muted);font-size:.6rem;text-transform:uppercase;letter-spacing:.1em}.metric .v{color:var(--text);font-family:'DM Mono',monospace;font-size:1.1rem;margin-top:.22rem}
    .signal{display:flex;justify-content:space-between;gap:1rem;padding:.62rem 0;border-bottom:1px solid #142031}.signal:last-child{border-bottom:0}.signal .name{color:#c5cfdb;font-size:.74rem}.signal .value{color:var(--cyan);font-family:'DM Mono',monospace;font-size:.68rem}
    .risk{padding:.68rem .78rem;border-left:2px solid var(--lime);background:rgba(163,230,53,.035);margin:.4rem 0;border-radius:0 9px 9px 0;color:#aeb9c8;font-size:.7rem;line-height:1.45}.risk.warn{border-left-color:var(--amber);background:rgba(245,158,11,.035)}
    .deck{border:1px solid var(--line);border-radius:24px;background:radial-gradient(circle at 88% 10%,rgba(34,211,238,.08),transparent 25%),linear-gradient(145deg,#0a111b,#080e17);min-height:500px;padding:2.25rem 2.5rem;position:relative;overflow:hidden}.deck:after{content:'';position:absolute;right:-120px;bottom:-120px;width:360px;height:360px;border:1px solid rgba(167,139,250,.12);border-radius:50%}
    .slide-count{color:var(--cyan);font-family:'DM Mono',monospace;font-size:.68rem}.slide-title{font-size:2.5rem;line-height:1;letter-spacing:-.06em;font-weight:800;margin:.55rem 0 .8rem;max-width:850px}.slide-copy{color:#a9b5c4;max-width:850px;line-height:1.75;font-size:.84rem}.slide-accent{color:var(--cyan)}
    .big-stat{font-family:'DM Mono',monospace;font-size:3rem;letter-spacing:-.08em;color:var(--cyan);margin:.7rem 0}.quote{border-left:2px solid var(--cyan);padding-left:1rem;color:#cbd5e1;font-size:.9rem;line-height:1.7;margin-top:1.2rem}.deck-footer{position:absolute;left:2.5rem;right:2.5rem;bottom:1.35rem;color:#536176;font-size:.61rem;letter-spacing:.08em;text-transform:uppercase}
    .footer-note{text-align:center;color:#5e6c80;font-size:.62rem;margin-top:2.2rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data / model
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return train_model(DATA_PATH)

if not DATA_PATH.exists():
    st.error("loan_data.csv was not found. Add the training dataset to the project root.")
    st.stop()

with st.spinner("Loading Loan Intelligence model…"):
    bundle, metrics = get_model()

BENCHMARK = pd.DataFrame(
    [
        ["Logistic Regression", 86.55, 0.64, 0.92, 0.75],
        ["KNN", 86.29, 0.64, 0.88, 0.74],
        ["Decision Tree", 89.28, 0.73, 0.82, 0.77],
        ["Random Forest", 89.48, 0.71, 0.88, 0.79],
        ["SVM", 88.02, 0.67, 0.92, 0.77],
        ["XGBoost", 92.87, 0.87, 0.80, 0.83],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"],
)

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈ NTI</div><div class="brand-title">Loan Intelligence</div><div class="brand-sub">Machine Learning Project</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Prediction", "Project Presentation", "Model Insights", "Prediction History"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="nav-label">System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="risk"><strong>XGBoost</strong><br>92.87% held-out test accuracy</div>'
        '<div class="risk"><strong>20 features</strong><br>44,990 processed rows</div>'
        '<div class="risk"><strong>SMOTETomek</strong><br>Applied to training data only</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Prediction workspace
# -----------------------------------------------------------------------------
if page == "Prediction":
    st.markdown(
        '<div class="hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK</div><h1>Loan <span>Intelligence.</span></h1><p>A polished decision-support interface for demonstrating binary loan-status classification using the project's benchmarked XGBoost model.</p><div class="hero-meta"><div class="pill"><strong>92.87%</strong> test accuracy</div><div class="pill"><strong>44,990</strong> processed rows</div><div class="pill"><strong>20</strong> model features</div><div class="pill"><strong>6</strong> benchmarked models</div></div></div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.18, .82], gap="large")
    with form_col:
        st.markdown('<div class="section-title">01 / Applicant profile</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 18, 80, 30, key="age")
                income = st.number_input("Annual income", 0.0, 2_000_000.0, 60_000.0, step=1_000.0, key="income")
                emp_exp = st.number_input("Employment experience (years)", 0, 60, 5, key="emp_exp")
                education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2, key="education")
                gender = st.selectbox("Gender", ["male", "female"], key="gender")
            with c2:
                home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], key="home")
                intent = st.selectbox("Loan intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], key="intent")
                loan_amount = st.number_input("Loan amount", 0.0, 500_000.0, 10_000.0, step=500.0, key="loan_amount")
                interest = st.number_input("Interest rate (%)", 0.0, 50.0, 10.0, step=.25, key="interest")
                loan_percent_income = st.number_input("Loan / income ratio", 0.0, 1.0, .20, step=.01, key="loan_percent_income")

        st.markdown('<div class="section-title">02 / Credit profile</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c3, c4, c5 = st.columns(3)
            with c3:
                credit_history = st.number_input("Credit history (years)", 0.0, 50.0, 5.0, step=.5, key="credit_history")
            with c4:
                credit_score = st.number_input("Credit score", 300, 850, 680, key="credit_score")
            with c5:
                previous_default = st.selectbox("Previous loan default", ["No", "Yes"], key="previous_default")

        st.markdown('<div class="section-title">03 / Run inference</div>', unsafe_allow_html=True)
        if st.button("Run loan prediction  →", type="primary", use_container_width=True):
            applicant = {
                "person_age": age,
                "person_income": income,
                "person_emp_exp": emp_exp,
                "person_gender": gender,
                "person_education": education,
                "person_home_ownership": home,
                "loan_intent": intent,
                "loan_amnt": loan_amount,
                "loan_int_rate": interest,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": credit_history,
                "credit_score": credit_score,
                "previous_loan_defaults_on_file": previous_default,
            }
            result = predict_one(bundle, applicant)
            st.session_state["result"] = result
            st.session_state["applicant"] = applicant
            history = st.session_state.setdefault("history", [])
            history.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Decision": result["label"],
                "Approval probability": f"{result['approval_probability']:.1%}",
                "Credit score": credit_score,
                "Loan amount": loan_amount,
            })
            st.session_state["history"] = history[:20]

    with result_col:
        st.markdown('<div class="section-title">Decision / model output</div>', unsafe_allow_html=True)
        if "result" not in st.session_state:
            st.markdown('<div class="decision"><div class="status mono">Awaiting applicant data</div><div class="label">READY</div><div class="prob">Run inference to generate a decision</div><div class="confidence-bar"><div class="confidence-fill" style="width:12%"></div></div><div class="decision-note">Educational ML demonstration • not financial advice.</div></div>', unsafe_allow_html=True)
        else:
            result = st.session_state["result"]
            probability = result["approval_probability"]
            approved = result["prediction"] == 1
            label_class = "approved" if approved else "rejected"
            st.markdown(f'<div class="decision"><div class="status mono">Predicted loan status</div><div class="label {label_class}">{result["label"].upper()}</div><div class="prob">{probability:.1%} approval probability</div><div class="confidence-bar"><div class="confidence-fill" style="width:{probability*100:.1f}%"></div></div><div class="decision-note">XGBoost • probability from model inference</div></div>', unsafe_allow_html=True)
            applicant = st.session_state.get("applicant", {})
            st.markdown('<div class="section-title">Decision signals</div>', unsafe_allow_html=True)
            flags = []
            if applicant.get("previous_loan_defaults_on_file") == "Yes":
                flags.append(("Previous default on file", "Historical risk signal", True))
            if applicant.get("loan_percent_income", 0) >= .40:
                flags.append(("High loan / income ratio", "Repayment burden signal", True))
            if applicant.get("credit_score", 850) < 600:
                flags.append(("Low credit score", "Credit risk signal", True))
            if not flags:
                flags.append(("No highlighted rule flags", "Model decision still depends on all features", False))
            for title, text, warning in flags:
                st.markdown(f'<div class="risk {"warn" if warning else ""}"><strong>{title}</strong><br>{text}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="metric-row"><div class="metric"><div class="k">Test accuracy</div><div class="v">{metrics["accuracy"]:.2%}</div></div><div class="metric"><div class="k">Test rows</div><div class="v">{metrics["test_rows"]:,}</div></div><div class="metric"><div class="k">Balanced train</div><div class="v">{metrics["train_rows_after_smotetomek"]:,}</div></div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Internal presentation deck
# -----------------------------------------------------------------------------
elif page == "Project Presentation":
    st.markdown('<div class="eyebrow mono">IN-APP PROJECT DECK</div><div class="page-title">Project Presentation</div><div class="page-sub">A compact presentation built into the product — from problem framing to model selection and deployment.</div>', unsafe_allow_html=True)

    slides = [
        ("01", "The problem", "Loan approval is a binary classification problem.", "Given applicant, loan, and credit-history attributes, the system learns a mapping to loan status: approved or rejected.", "The objective is not to replace a lender. It is to demonstrate a complete, reproducible ML workflow from raw data to inference."),
        ("02", "The data", "44,990 rows • 20 processed features", "The project combines applicant profile, financial attributes, loan characteristics, and credit-history signals. The target is loan_status.", "Examples include income, loan amount, interest rate, loan-to-income ratio, credit score, home ownership, loan intent, and previous defaults."),
        ("03", "The pipeline", "Clean → encode → split → scale → balance → train", "The notebook workflow is reproduced in reusable Python code. Identifiers are removed, selected numeric outliers are clipped, categorical variables are encoded, and the data is split before training transformations.", "RobustScaler is fitted on the training split and SMOTETomek is applied only to the training data."),
        ("04", "Model benchmark", "Six classifiers. One held-out test set.", "Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost were compared using the same benchmark setup.", "XGBoost reached 92.87% test accuracy, 0.87 class-1 precision, and 0.83 class-1 F1. Logistic Regression and SVM reached higher class-1 recall at 0.92."),
        ("05", "Why XGBoost?", "Best overall benchmark trade-off", "XGBoost was selected because it led the benchmark on test accuracy, class-1 precision, and class-1 F1.", "This is a benchmark selection, not a claim of universal superiority. Recall matters too, especially when false negatives have a higher cost."),
        ("06", "The product", "From notebook to interactive inference", "The Streamlit layer turns the trained workflow into an interactive demonstration where an applicant profile can be submitted and scored in real time.", "The interface exposes the prediction, approval probability, highlighted rule-based signals, and core model metrics."),
        ("07", "Limitations", "A strong portfolio project is honest about its boundaries.", "The dataset is a learning dataset, the benchmark is not a production credit policy, and the system has not been audited for fairness, calibration, or dataset shift.", "The current model output should be treated as an educational prediction — never as the sole basis for a real financial decision."),
        ("08", "Next level", "Production hardening roadmap", "A production-oriented version could add a versioned model artifact, FastAPI serving, SHAP explanations, calibration, fairness evaluation, Docker, MLflow, monitoring, and CI/CD.", "The architecture already separates the reusable ML logic from the Streamlit presentation layer, making these upgrades easier to introduce."),
        ("09", "Takeaway", "A complete ML system is more than a model.", "The project demonstrates problem framing, data preparation, class balancing, model benchmarking, honest evaluation, reusable inference code, testing, and a user-facing interface.", "That end-to-end path is the core engineering story behind Loan Intelligence."),
    ]

    if "slide" not in st.session_state:
        st.session_state["slide"] = 0
    idx = st.session_state["slide"]
    code, title, headline, copy, note = slides[idx]

    st.markdown(
        f'<div class="deck"><div class="slide-count">SLIDE {code} / {len(slides):02d}</div><div class="slide-title">{title}<br><span class="slide-accent">{headline}</span></div><div class="slide-copy">{copy}</div><div class="quote">{note}</div><div class="deck-footer">LOAN INTELLIGENCE • NTI MACHINE LEARNING • {code}</div></div>',
        unsafe_allow_html=True,
    )
    st.progress((idx + 1) / len(slides))
    prev_col, spacer, next_col = st.columns([1, 3, 1])
    with prev_col:
        if st.button("← Previous", use_container_width=True, disabled=idx == 0):
            st.session_state["slide"] = max(0, idx - 1)
            st.rerun()
    with next_col:
        if st.button("Next →", use_container_width=True, disabled=idx == len(slides) - 1):
            st.session_state["slide"] = min(len(slides) - 1, idx + 1)
            st.rerun()

# -----------------------------------------------------------------------------
# Model insights
# -----------------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown('<div class="eyebrow mono">EVALUATION / EXPLAINABILITY</div><div class="page-title">Model Insights</div><div class="page-sub">Benchmark evidence and model-level signals from the project notebook and reusable training pipeline.</div>', unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    for col, value, label in [
        (a, "92.87%", "XGBoost test accuracy"),
        (b, "0.87", "Class-1 precision"),
        (c, "0.83", "Class-1 F1"),
        (d, "0.80", "Class-1 recall"),
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="card-k">{label}</div><div class="card-v">{value}</div><div class="card-note">Held-out benchmark result</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown('<div class="section-title">Model benchmark</div>', unsafe_allow_html=True)
        chart_df = BENCHMARK.set_index("Model")[["Accuracy"]] / 100
        st.bar_chart(chart_df, height=330)
        st.caption("Accuracy comparison from the same held-out test benchmark.")
    with right:
        st.markdown('<div class="section-title">Benchmark table</div>', unsafe_allow_html=True)
        display = BENCHMARK.copy()
        display["Accuracy"] = display["Accuracy"].map(lambda x: f"{x:.2f}%")
        for col in ["Precision", "Recall", "F1"]:
            display[col] = display[col].map(lambda x: f"{x:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True, height=330)

    st.markdown('<div class="section-title">Feature importance / model signal</div>', unsafe_allow_html=True)
    importance = pd.DataFrame({"Feature": bundle.feature_columns, "Importance": bundle.model.feature_importances_}).sort_values("Importance", ascending=False).head(10).set_index("Feature")
    st.bar_chart(importance, height=330)
    st.caption("XGBoost feature importance is shown as a model diagnostic; it is not causal evidence and does not replace explainability or fairness analysis.")

    st.markdown('<div class="section-title">What the benchmark says</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><div class="step"><div class="step-no">01</div><div><h4>Strongest overall benchmark</h4><p>XGBoost leads test accuracy, class-1 precision, and class-1 F1 in this project.</p></div></div><div class="step"><div class="step-no">02</div><div><h4>Recall trade-off</h4><p>Logistic Regression and SVM reach 0.92 class-1 recall, higher than XGBoost’s 0.80. Model choice depends on the cost of false negatives versus false positives.</p></div></div><div class="step"><div class="step-no">03</div><div><h4>Overfitting awareness</h4><p>The Decision Tree reached 100% training accuracy but 89.28% test accuracy, a visible generalization gap.</p></div></div></div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Prediction history
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="eyebrow mono">SESSION / INFERENCE LOG</div><div class="page-title">Prediction History</div><div class="page-sub">Recent predictions from this browser session. Nothing here is persisted to a database.</div>', unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if not history:
        st.markdown('<div class="card"><div class="card-k">No predictions yet</div><div class="card-v">—</div><div class="card-note">Run a prediction from the Prediction workspace and it will appear here.</div></div>', unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("Clear session history"):
            st.session_state["history"] = []
            st.rerun()

st.markdown('<div class="footer-note">LOAN INTELLIGENCE • NTI MACHINE LEARNING • EDUCATIONAL DEMONSTRATION ONLY • NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)
