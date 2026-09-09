from pathlib import Path

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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root{--bg:#060910;--panel:#0b111c;--line:#1c2a3d;--text:#f4f7fb;--muted:#8391a6;--cyan:#22d3ee;--lime:#a3e635;--danger:#fb7185}
    html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:var(--bg);color:var(--text)}
    [data-testid="stHeader"]{background:rgba(6,9,16,.9)}[data-testid="stSidebar"]{background:#080d16;border-right:1px solid var(--line)}
    .block-container{max-width:1380px;padding:1.6rem 2.3rem 3rem}#MainMenu,footer{visibility:hidden}
    .mono{font-family:'DM Mono',monospace;letter-spacing:.08em}.eyebrow{color:var(--cyan);font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase}
    .brand{padding:.4rem .2rem 1.2rem}.brand-mark{color:var(--cyan);font-size:1.25rem;font-weight:800;letter-spacing:-.05em}.brand-title{font-size:1.02rem;font-weight:800;margin-top:.2rem}.brand-sub{color:var(--muted);font-size:.72rem;margin-top:.25rem}
    .hero{position:relative;overflow:hidden;min-height:230px;padding:2.1rem 2.25rem;border:1px solid var(--line);border-radius:24px;background:radial-gradient(circle at 84% 18%,rgba(34,211,238,.13),transparent 26%),radial-gradient(circle at 72% 110%,rgba(167,139,250,.11),transparent 32%),linear-gradient(135deg,#0a111d,#09101a 58%,#0b1020);margin-bottom:1.25rem}
    .hero:after{content:'';position:absolute;right:-90px;top:-110px;width:300px;height:300px;border:1px solid rgba(34,211,238,.14);border-radius:50%;box-shadow:0 0 0 28px rgba(34,211,238,.025),0 0 0 56px rgba(34,211,238,.015)}
    .hero h1{margin:.35rem 0 0;font-size:clamp(2.35rem,5vw,4.6rem);line-height:.94;letter-spacing:-.065em;font-weight:800}.hero h1 span{color:var(--cyan)}.hero p{color:#9aa8ba;max-width:690px;margin:.9rem 0 0;font-size:.98rem;line-height:1.65}
    .hero-meta{margin-top:1.35rem;display:flex;gap:.55rem;flex-wrap:wrap}.pill{border:1px solid var(--line);background:rgba(255,255,255,.025);color:#aeb9c8;padding:.38rem .62rem;border-radius:999px;font-size:.69rem}.pill strong{color:var(--text)}
    .section-title{margin:1.1rem 0 .65rem;font-size:.76rem;text-transform:uppercase;letter-spacing:.13em;color:#8fa0b5;font-weight:800}
    label,.stNumberInput label,.stSelectbox label{color:#b7c1ce!important;font-size:.73rem!important;font-weight:600!important}
    input,[data-baseweb="select"]>div{background:#080e18!important;border-color:#1c2b3e!important;color:var(--text)!important;border-radius:10px!important}
    input:focus{border-color:rgba(34,211,238,.65)!important;box-shadow:0 0 0 1px rgba(34,211,238,.15)!important}.stButton>button{border:1px solid rgba(34,211,238,.45);background:linear-gradient(135deg,#0e2530,#0b1721);color:#dffbff;border-radius:11px;min-height:48px;font-weight:800;box-shadow:0 0 28px rgba(34,211,238,.07)}.stButton>button:hover{border-color:var(--cyan);color:white}
    .decision{min-height:270px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border:1px solid var(--line);border-radius:20px;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.09),transparent 48%),var(--panel);padding:1.5rem}.decision.ready{background:radial-gradient(circle at 50% 0%,rgba(167,139,250,.08),transparent 45%),var(--panel)}
    .decision .status{font-size:.68rem;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:800}.decision .label{margin-top:.45rem;font-size:2.4rem;line-height:1;font-weight:800;letter-spacing:-.055em}.decision .approved{color:var(--lime)}.decision .rejected{color:var(--danger)}.prob{font-family:'DM Mono',monospace;margin-top:.8rem;font-size:1.12rem;color:#d6e3ef}.confidence-bar{width:82%;height:6px;background:#172232;border-radius:99px;overflow:hidden;margin-top:1rem}.confidence-fill{height:100%;background:var(--cyan);border-radius:99px}.decision-note{color:var(--muted);font-size:.68rem;margin-top:.75rem}
    .metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-top:.75rem}.metric{border:1px solid var(--line);background:#090f19;border-radius:14px;padding:.9rem 1rem}.metric .k{color:var(--muted);font-size:.66rem;text-transform:uppercase;letter-spacing:.1em}.metric .v{color:var(--text);font-family:'DM Mono',monospace;font-size:1.18rem;margin-top:.25rem}
    .signal{display:flex;justify-content:space-between;gap:1rem;padding:.62rem 0;border-bottom:1px solid #142031}.signal:last-child{border-bottom:0}.signal .name{color:#c5cfdb;font-size:.76rem}.signal .value{color:var(--cyan);font-family:'DM Mono',monospace;font-size:.72rem}
    .risk{padding:.7rem .8rem;border-left:2px solid var(--lime);background:rgba(163,230,53,.035);margin:.4rem 0;border-radius:0 9px 9px 0;color:#aeb9c8;font-size:.73rem}.risk.warn{border-left-color:#f59e0b;background:rgba(245,158,11,.035)}.footer-note{text-align:center;color:#647286;font-size:.66rem;margin-top:2rem}
    </style>
    """, unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def get_model():
    return train_model(DATA_PATH)

if not DATA_PATH.exists():
    st.error("loan_data.csv was not found. Add the training dataset to the project root.")
    st.stop()

with st.spinner("Loading Loan Intelligence model…"):
    bundle, metrics = get_model()

with st.sidebar:
    st.markdown("""
    <div class="brand"><div class="brand-mark">◈ NTI</div><div class="brand-title">Loan Intelligence</div><div class="brand-sub">Machine Learning Project</div></div>
    <div class="section-title">Model status</div>
    <div class="risk"><strong>XGBoost</strong><br>Benchmark accuracy: 92.87%</div>
    <div class="risk"><strong>Binary classification</strong><br>Loan status prediction</div>
    <div class="section-title">Pipeline</div>
    <div class="risk">01 — Clean & encode</div><div class="risk">02 — Robust scaling</div><div class="risk">03 — SMOTETomek</div><div class="risk">04 — XGBoost</div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK</div><h1>Loan <span>Intelligence.</span></h1><p>A decision-support interface for predicting loan status from applicant, credit, and loan attributes — powered by the project's XGBoost benchmark model.</p><div class="hero-meta"><div class="pill"><strong>92.87%</strong> test accuracy</div><div class="pill"><strong>44,990</strong> processed rows</div><div class="pill"><strong>20</strong> model features</div><div class="pill"><strong>SMOTETomek</strong> class balancing</div></div></div>
""", unsafe_allow_html=True)

form_col, result_col = st.columns([1.18, .82], gap="large")

with form_col:
    st.markdown('<div class="section-title">01 / Applicant profile</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 80, 30)
            income = st.number_input("Annual income", 0.0, 2_000_000.0, 60_000.0, step=1_000.0)
            emp_exp = st.number_input("Employment experience (years)", 0, 60, 5)
            education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2)
            gender = st.selectbox("Gender", ["male", "female"])
        with c2:
            home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            intent = st.selectbox("Loan intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
            loan_amount = st.number_input("Loan amount", 0.0, 500_000.0, 10_000.0, step=500.0)
            interest = st.number_input("Interest rate (%)", 0.0, 50.0, 10.0, step=.25)
            loan_percent_income = st.number_input("Loan / income ratio", 0.0, 1.0, .20, step=.01)

    st.markdown('<div class="section-title">02 / Credit profile</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c3, c4, c5 = st.columns(3)
        with c3: credit_history = st.number_input("Credit history (years)", 0.0, 50.0, 5.0, step=.5)
        with c4: credit_score = st.number_input("Credit score", 300, 850, 680)
        with c5: previous_default = st.selectbox("Previous loan default", ["No", "Yes"])

    st.markdown('<div class="section-title">03 / Run inference</div>', unsafe_allow_html=True)
    if st.button("Run loan prediction  →", type="primary", use_container_width=True):
        applicant = {"person_age": age, "person_income": income, "person_emp_exp": emp_exp, "person_gender": gender, "person_education": education, "person_home_ownership": home, "loan_intent": intent, "loan_amnt": loan_amount, "loan_int_rate": interest, "loan_percent_income": loan_percent_income, "cb_person_cred_hist_length": credit_history, "credit_score": credit_score, "previous_loan_defaults_on_file": previous_default}
        st.session_state["result"] = predict_one(bundle, applicant)
        st.session_state["applicant"] = applicant

with result_col:
    st.markdown('<div class="section-title">Decision / model output</div>', unsafe_allow_html=True)
    if "result" not in st.session_state:
        st.markdown('<div class="decision ready"><div class="status mono">Awaiting applicant data</div><div class="label">READY</div><div class="prob">Run inference to generate a decision</div><div class="confidence-bar"><div class="confidence-fill" style="width:12%"></div></div><div class="decision-note">Educational ML demonstration.</div></div>', unsafe_allow_html=True)
    else:
        result = st.session_state["result"]
        probability = result["approval_probability"]
        approved = result["prediction"] == 1
        label_class = "approved" if approved else "rejected"
        st.markdown(f'<div class="decision"><div class="status mono">Predicted loan status</div><div class="label {label_class}">{result["label"].upper()}</div><div class="prob">{probability:.1%} approval probability</div><div class="confidence-bar"><div class="confidence-fill" style="width:{probability*100:.1f}%"></div></div><div class="decision-note">XGBoost • probability from model inference</div></div>', unsafe_allow_html=True)
        applicant = st.session_state.get("applicant", {})
        st.markdown('<div class="section-title">Decision signals</div>', unsafe_allow_html=True)
        flags = []
        if applicant.get("previous_loan_defaults_on_file") == "Yes": flags.append(("Previous default on file", "Historical risk signal", True))
        if applicant.get("loan_percent_income", 0) >= .40: flags.append(("High loan / income ratio", "Repayment burden signal", True))
        if applicant.get("credit_score", 850) < 600: flags.append(("Low credit score", "Credit risk signal", True))
        if not flags: flags.append(("No highlighted rule flags", "Model decision still depends on all features", False))
        for title, text, warning in flags:
            st.markdown(f'<div class="risk {"warn" if warning else ""}"><strong>{title}</strong><br>{text}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="metric-row"><div class="metric"><div class="k">Test accuracy</div><div class="v">{metrics["accuracy"]:.2%}</div></div><div class="metric"><div class="k">Test rows</div><div class="v">{metrics["test_rows"]:,}</div></div><div class="metric"><div class="k">Balanced train</div><div class="v">{metrics["train_rows_after_smotetomek"]:,}</div></div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer-note">LOAN INTELLIGENCE • NTI MACHINE LEARNING • EDUCATIONAL DEMONSTRATION ONLY • NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)
