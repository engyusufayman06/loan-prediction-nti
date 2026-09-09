from pathlib import Path
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from src.model import MODEL_INPUT_COLUMNS, predict_batch, predict_one, train_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_data.csv"
NTI_SITE = "https://www.nti.sci.eg/"
NTI_LOGO = "https://www.nti.sci.eg/favicon.ico"
TEAM = [
    "Yusuf Ayman Tolba",
    "Mohamed Reda Hussein",
    "Abdelrahman Mohamed Ahmed",
    "Abdelmoniem Ibrahim Abdelmoniem",
]

st.set_page_config(
    page_title="Loan Intelligence | NTI ML",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#04070d;--panel:#080e17;--panel2:#0c1420;--line:#182638;--text:#f5f7fb;--muted:#8291a5;--cyan:#22d3ee;--lime:#a3e635;--purple:#a78bfa;--danger:#fb7185;--amber:#f59e0b}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(4,7,13,.88)}[data-testid="stSidebar"]{background:#060b12;border-right:1px solid var(--line)}
.block-container{max-width:1500px;padding:1.25rem 2rem 4rem}#MainMenu,footer{visibility:hidden}
.mono{font-family:'DM Mono',monospace;letter-spacing:.08em}.eyebrow{color:var(--cyan);font-size:.65rem;font-weight:800;letter-spacing:.2em;text-transform:uppercase}
.brand{padding:.3rem .1rem 1rem}.brand-row{display:flex;align-items:center;gap:.7rem}.brand-logo{width:38px;height:38px;border-radius:10px;object-fit:contain;background:white;padding:4px}.brand-mark{color:var(--cyan);font-weight:800}.brand-title{font-size:1rem;font-weight:800}.brand-sub{color:var(--muted);font-size:.65rem;margin-top:.18rem}.nav-label{font-size:.61rem;color:#607086;text-transform:uppercase;letter-spacing:.17em;font-weight:800;margin:1rem 0 .45rem}
.team-mini{color:#9aa8b9;font-size:.65rem;line-height:1.65;border-top:1px solid var(--line);padding-top:.8rem}.team-mini strong{color:#dfe8f2}
.hero{position:relative;overflow:hidden;min-height:265px;padding:2.5rem;border:1px solid var(--line);border-radius:28px;background:radial-gradient(circle at 90% 10%,rgba(34,211,238,.17),transparent 25%),radial-gradient(circle at 72% 110%,rgba(167,139,250,.13),transparent 31%),linear-gradient(135deg,#08111d,#07101a 58%,#0d1020);margin-bottom:1.3rem}.hero:after{content:'';position:absolute;right:-90px;top:-130px;width:360px;height:360px;border:1px solid rgba(34,211,238,.16);border-radius:50%;box-shadow:0 0 0 32px rgba(34,211,238,.025),0 0 0 68px rgba(34,211,238,.014)}
.hero h1{margin:.35rem 0 0;font-size:clamp(2.7rem,5.8vw,5.6rem);line-height:.9;letter-spacing:-.075em;font-weight:800}.hero h1 span{color:var(--cyan)}.hero p{color:#a1afbf;max-width:850px;margin:1rem 0 0;font-size:.9rem;line-height:1.75}.hero-meta{margin-top:1.4rem;display:flex;gap:.5rem;flex-wrap:wrap}.pill{border:1px solid var(--line);background:rgba(255,255,255,.025);color:#aeb9c8;padding:.42rem .68rem;border-radius:999px;font-size:.65rem}.pill strong{color:var(--text)}
.page-title{font-size:2.35rem;font-weight:800;letter-spacing:-.06em;margin:.1rem 0 .25rem}.page-sub{color:var(--muted);font-size:.8rem;margin-bottom:1.25rem}.section-title{margin:1.15rem 0 .65rem;font-size:.64rem;text-transform:uppercase;letter-spacing:.18em;color:#8fa0b5;font-weight:800}
.card{border:1px solid var(--line);background:linear-gradient(145deg,#09111b,#070d16);border-radius:18px;padding:1.15rem 1.2rem;height:100%}.card-k{color:#718198;font-size:.6rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800}.card-v{font-family:'DM Mono',monospace;font-size:1.55rem;margin-top:.3rem}.card-note{color:var(--muted);font-size:.66rem;margin-top:.3rem;line-height:1.5}
.step{display:flex;gap:.85rem;align-items:flex-start;padding:.82rem 0;border-bottom:1px solid #132031}.step:last-child{border-bottom:0}.step-no{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.68rem;border:1px solid #214051;border-radius:8px;padding:.32rem .42rem}.step h4{margin:0;color:#e8edf4;font-size:.77rem}.step p{margin:.18rem 0 0;color:var(--muted);font-size:.67rem;line-height:1.5}
label,.stNumberInput label,.stSelectbox label,.stTextInput label{color:#b7c1ce!important;font-size:.69rem!important;font-weight:600!important}input,[data-baseweb="select"]>div{background:#070d16!important;border-color:#1a2a3d!important;color:var(--text)!important;border-radius:10px!important}.stButton>button{border:1px solid rgba(34,211,238,.35);background:linear-gradient(135deg,#0c202a,#09151e);color:#dffbff;border-radius:11px;min-height:44px;font-weight:800}.stButton>button:hover{border-color:var(--cyan);color:white}.stDownloadButton>button{border-radius:11px}
.decision{min-height:280px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border:1px solid var(--line);border-radius:20px;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.09),transparent 48%),var(--panel);padding:1.5rem}.decision .status{font-size:.61rem;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:800}.decision .label{margin-top:.45rem;font-size:2.55rem;line-height:1;font-weight:800;letter-spacing:-.055em}.approved{color:var(--lime)}.rejected{color:var(--danger)}.prob{font-family:'DM Mono',monospace;margin-top:.8rem;font-size:1.05rem;color:#d6e3ef}.confidence-bar{width:82%;height:7px;background:#172232;border-radius:99px;overflow:hidden;margin-top:1rem}.confidence-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));border-radius:99px}.decision-note{color:var(--muted);font-size:.63rem;margin-top:.75rem}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem;margin-top:.75rem}.metric{border:1px solid var(--line);background:#090f19;border-radius:13px;padding:.82rem .9rem}.metric .k{color:var(--muted);font-size:.58rem;text-transform:uppercase;letter-spacing:.1em}.metric .v{color:var(--text);font-family:'DM Mono',monospace;font-size:1.05rem;margin-top:.22rem}
.risk{padding:.68rem .78rem;border-left:2px solid var(--lime);background:rgba(163,230,53,.035);margin:.4rem 0;border-radius:0 9px 9px 0;color:#aeb9c8;font-size:.69rem;line-height:1.45}.risk.warn{border-left-color:var(--amber);background:rgba(245,158,11,.035)}.range-note{font-size:.59rem;color:#66768b;margin-top:-.35rem;margin-bottom:.5rem}
.deck{border:1px solid var(--line);border-radius:26px;background:radial-gradient(circle at 88% 8%,rgba(34,211,238,.09),transparent 25%),linear-gradient(145deg,#0a121d,#070d16);min-height:535px;padding:2.5rem;position:relative;overflow:hidden}.deck:before{content:'LOAN\AINTELLIGENCE';white-space:pre;position:absolute;right:2rem;top:1.2rem;text-align:right;color:rgba(255,255,255,.035);font-size:4.8rem;font-weight:800;line-height:.8;letter-spacing:-.08em}.slide-count{color:var(--cyan);font-family:'DM Mono',monospace;font-size:.65rem}.slide-title{font-size:2.65rem;line-height:1.02;letter-spacing:-.065em;font-weight:800;margin:.6rem 0 .9rem;max-width:950px}.slide-copy{color:#a9b5c4;max-width:900px;line-height:1.8;font-size:.83rem}.slide-accent{color:var(--cyan)}.big-stat{font-family:'DM Mono',monospace;font-size:4rem;letter-spacing:-.08em;color:var(--cyan);margin:.8rem 0}.quote{border-left:2px solid var(--cyan);padding-left:1rem;color:#cbd5e1;font-size:.87rem;line-height:1.7;margin-top:1.3rem;max-width:900px}.deck-footer{position:absolute;left:2.5rem;right:2.5rem;bottom:1.25rem;color:#536176;font-size:.58rem;letter-spacing:.09em;text-transform:uppercase}
.team-card{border:1px solid var(--line);border-radius:16px;background:#090f18;padding:1rem}.team-number{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.65rem}.team-name{font-size:.82rem;font-weight:700;margin-top:.35rem}.team-role{color:var(--muted);font-size:.62rem;margin-top:.25rem}.footer-note{text-align:center;color:#536176;font-size:.6rem;margin-top:2.3rem;letter-spacing:.06em}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model():
    return train_model(DATA_PATH)

@st.cache_data(show_spinner=False)
def get_reference_data():
    return pd.read_csv(DATA_PATH)

if not DATA_PATH.exists():
    st.error("loan_data.csv was not found in the project root.")
    st.stop()

with st.spinner("Loading Loan Intelligence model…"):
    bundle, metrics = get_model()
reference = get_reference_data()

BENCHMARK = pd.DataFrame([
    ["Logistic Regression", 86.55, .64, .92, .75],
    ["KNN", 86.29, .64, .88, .74],
    ["Decision Tree", 89.28, .73, .82, .77],
    ["Random Forest", 89.48, .71, .88, .79],
    ["SVM", 88.02, .67, .92, .77],
    ["XGBoost", 92.87, .87, .80, .83],
], columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

# Runtime ranges are derived from the supplied CSV, so every numeric input is visibly bounded by the project data.
RANGES = {
    "age": (int(max(18, reference.person_age.min())), int(min(80, reference.person_age.max()))),
    "income": (float(reference.person_income.min()), float(reference.person_income.max())),
    "emp_exp": (int(max(0, reference.person_emp_exp.min())), int(min(60, reference.person_emp_exp.max()))),
    "loan_amount": (float(reference.loan_amnt.min()), float(reference.loan_amnt.max())),
    "interest": (float(reference.loan_int_rate.min()), float(reference.loan_int_rate.max())),
    "loan_percent_income": (float(reference.loan_percent_income.min()), float(min(1, reference.loan_percent_income.max()))),
    "credit_history": (float(reference.cb_person_cred_hist_length.min()), float(reference.cb_person_cred_hist_length.max())),
    "credit_score": (int(reference.credit_score.min()), int(reference.credit_score.max())),
}

with st.sidebar:
    st.markdown(f'''<div class="brand"><div class="brand-row"><img class="brand-logo" src="{NTI_LOGO}" onerror="this.style.display='none'"><div><div class="brand-mark">NTI / ML</div><div class="brand-title">Loan Intelligence</div><div class="brand-sub">End-to-end classification system</div></div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Prediction Studio", "Batch Lab", "Project Presentation", "Model Insights", "Team & About"], label_visibility="collapsed")
    st.markdown('<div class="nav-label">Live system</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk"><strong>XGBoost</strong><br>{metrics["accuracy"]:.2%} held-out test accuracy</div><div class="risk"><strong>{len(reference):,} source rows</strong><br>{len(bundle.feature_columns)} encoded model features</div><div class="risk"><strong>Training balance</strong><br>{metrics["train_rows_after_smotetomek"]:,} rows after SMOTETomek</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="team-mini">' + '<br>'.join(f'<strong>{i+1}.</strong> {n}' for i,n in enumerate(TEAM)) + '</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:.8rem;font-size:.6rem"><a href="{NTI_SITE}" target="_blank" style="color:#22d3ee;text-decoration:none">Official NTI website ↗</a></div>', unsafe_allow_html=True)

# ----------------------------- Prediction Studio -----------------------------
if page == "Prediction Studio":
    st.markdown(f'''<div class="hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK / TEAM PROJECT</div><h1>Loan <span>Intelligence.</span></h1><p>Interactive binary loan-status prediction powered by the same preprocessing and XGBoost benchmark used in the project notebook. Every numeric control is bounded by the reference dataset.</p><div class="hero-meta"><div class="pill"><strong>{metrics["accuracy"]:.2%}</strong> test accuracy</div><div class="pill"><strong>{len(reference):,}</strong> source rows</div><div class="pill"><strong>{len(bundle.feature_columns)}</strong> encoded features</div><div class="pill"><strong>6</strong> models benchmarked</div></div></div>''', unsafe_allow_html=True)

    form_col, result_col = st.columns([1.18, .82], gap="large")
    with form_col:
        st.markdown('<div class="section-title">01 / Applicant profile — all model inputs</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1,c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=RANGES["age"][0], max_value=RANGES["age"][1], value=min(30,RANGES["age"][1]), step=1)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["age"][0]} → {RANGES["age"][1]}</div>', unsafe_allow_html=True)
                income = st.number_input("Annual income", min_value=RANGES["income"][0], max_value=RANGES["income"][1], value=min(60000.,RANGES["income"][1]), step=1000.)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["income"][0]:,.0f} → {RANGES["income"][1]:,.0f}</div>', unsafe_allow_html=True)
                emp_exp = st.number_input("Employment experience (years)", min_value=RANGES["emp_exp"][0], max_value=RANGES["emp_exp"][1], value=min(5,RANGES["emp_exp"][1]), step=1)
                st.markdown(f'<div class="range-note">Model validation: ≤ 60 years • dataset range: {RANGES["emp_exp"][0]} → {RANGES["emp_exp"][1]}</div>', unsafe_allow_html=True)
                education = st.selectbox("Education", ["High School","Associate","Bachelor","Master","Doctorate"], index=2)
                gender = st.selectbox("Gender", ["male","female"])
            with c2:
                home = st.selectbox("Home ownership", ["RENT","OWN","MORTGAGE","OTHER"])
                intent = st.selectbox("Loan intent", ["EDUCATION","MEDICAL","VENTURE","PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"])
                loan_amount = st.number_input("Loan amount", min_value=RANGES["loan_amount"][0], max_value=RANGES["loan_amount"][1], value=min(10000.,RANGES["loan_amount"][1]), step=500.)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["loan_amount"][0]:,.0f} → {RANGES["loan_amount"][1]:,.0f}</div>', unsafe_allow_html=True)
                interest = st.number_input("Interest rate (%)", min_value=RANGES["interest"][0], max_value=RANGES["interest"][1], value=min(10.,RANGES["interest"][1]), step=.25)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["interest"][0]:.2f} → {RANGES["interest"][1]:.2f}</div>', unsafe_allow_html=True)
                loan_percent_income = st.number_input("Loan / income ratio", min_value=RANGES["loan_percent_income"][0], max_value=RANGES["loan_percent_income"][1], value=min(.20,RANGES["loan_percent_income"][1]), step=.01)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["loan_percent_income"][0]:.2f} → {RANGES["loan_percent_income"][1]:.2f}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">02 / Credit profile</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c3,c4,c5 = st.columns(3)
            with c3:
                credit_history = st.number_input("Credit history (years)", min_value=RANGES["credit_history"][0], max_value=RANGES["credit_history"][1], value=min(5.,RANGES["credit_history"][1]), step=.5)
            with c4:
                credit_score = st.number_input("Credit score", min_value=RANGES["credit_score"][0], max_value=RANGES["credit_score"][1], value=min(680,RANGES["credit_score"][1]), step=1)
            with c5:
                previous_default = st.selectbox("Previous loan default", ["No","Yes"])
            st.markdown(f'<div class="range-note">Credit history: {RANGES["credit_history"][0]:.1f} → {RANGES["credit_history"][1]:.1f} years • Credit score: {RANGES["credit_score"][0]} → {RANGES["credit_score"][1]}</div>', unsafe_allow_html=True)

        if st.button("Run Loan Intelligence  →", type="primary", use_container_width=True):
            applicant = {
                "person_age": age, "person_income": income, "person_emp_exp": emp_exp,
                "person_gender": gender, "person_education": education,
                "person_home_ownership": home, "loan_intent": intent,
                "loan_amnt": loan_amount, "loan_int_rate": interest,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": credit_history, "credit_score": credit_score,
                "previous_loan_defaults_on_file": previous_default,
            }
            result = predict_one(bundle, applicant)
            st.session_state["result"] = result
            st.session_state["applicant"] = applicant
            history = st.session_state.setdefault("history", [])
            history.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Status": result["label"],
                "Approval probability": f'{result["approval_probability"]:.1%}',
                "Credit score": credit_score,
                "Loan amount": loan_amount,
            })
            st.session_state["history"] = history[:20]
            st.rerun()

    with result_col:
        st.markdown('<div class="section-title">03 / Decision intelligence</div>', unsafe_allow_html=True)
        result = st.session_state.get("result")
        if not result:
            st.markdown('<div class="decision"><div class="status mono">Awaiting applicant data</div><div class="label">READY</div><div class="prob">Enter the full applicant profile and run inference</div><div class="confidence-bar"><div class="confidence-fill" style="width:8%"></div></div><div class="decision-note">Educational ML demonstration • not financial advice.</div></div>', unsafe_allow_html=True)
        else:
            probability = result["approval_probability"]
            approved = result["prediction"] == 1
            label_class = "approved" if approved else "rejected"
            st.markdown(f'<div class="decision"><div class="status mono">Predicted loan status</div><div class="label {label_class}">{result["label"].upper()}</div><div class="prob">{probability:.1%} approval probability</div><div class="confidence-bar"><div class="confidence-fill" style="width:{probability*100:.1f}%"></div></div><div class="decision-note">XGBoost • probability from model inference</div></div>', unsafe_allow_html=True)
            applicant = st.session_state.get("applicant", {})
            st.markdown('<div class="section-title">Signals to inspect</div>', unsafe_allow_html=True)
            flags=[]
            if applicant.get("previous_loan_defaults_on_file") == "Yes": flags.append(("Previous default on file","Historical risk signal",True))
            if applicant.get("loan_percent_income",0) >= .40: flags.append(("High loan / income ratio","Repayment burden signal",True))
            if applicant.get("credit_score",850) < 600: flags.append(("Low credit score","Credit risk signal",True))
            if not flags: flags.append(("No highlighted rule flags","The model still uses all encoded features",False))
            for title,text,warning in flags:
                st.markdown(f'<div class="risk {"warn" if warning else ""}"><strong>{title}</strong><br>{text}</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="metric-row"><div class="metric"><div class="k">Test accuracy</div><div class="v">{metrics["accuracy"]:.2%}</div></div><div class="metric"><div class="k">Test rows</div><div class="v">{metrics["test_rows"]:,}</div></div><div class="metric"><div class="k">Balanced train</div><div class="v">{metrics["train_rows_after_smotetomek"]:,}</div></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Inference contract</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-k">Exactly 13 raw applicant fields</div><div class="card-note">The UI collects every raw field required by the trained inference contract. Categorical encoding, IQR clipping, scaling and feature alignment remain inside the reusable model layer.</div></div>',unsafe_allow_html=True)

# ----------------------------- Batch Lab -----------------------------
elif page == "Batch Lab":
    st.markdown('<div class="eyebrow mono">BATCH INFERENCE / CSV WORKSPACE</div><div class="page-title">Batch Lab</div><div class="page-sub">Upload a CSV containing the same raw applicant fields used by the notebook, score every row, inspect results, and download the predictions.</div>', unsafe_allow_html=True)

    template = reference[MODEL_INPUT_COLUMNS + (["loan_id","loan_status"] if "loan_id" in reference.columns and "loan_status" in reference.columns else [])].head(5)
    st.download_button("Download CSV template", data=template.to_csv(index=False).encode("utf-8"), file_name="loan_intelligence_template.csv", mime="text/csv", type="secondary")
    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"], help="Required raw columns are shown below. The target loan_status is optional and is used only for evaluation.")

    with st.expander("Required CSV schema", expanded=False):
        schema = pd.DataFrame({
            "Column": MODEL_INPUT_COLUMNS,
            "Role": ["numeric","numeric","categorical","numeric","categorical","numeric","numeric","numeric","numeric","numeric","categorical","categorical","ordinal categorical"],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)
        st.caption("Optional columns: loan_id and loan_status. If loan_status is supplied, the app also reports row-level correctness.")

    if uploaded is None:
        st.markdown('<div class="card" style="margin-top:1rem"><div class="card-k">No file uploaded</div><div class="card-v">CSV → SCORE → DOWNLOAD</div><div class="card-note">Use the template button above, edit the applicant rows, then upload the CSV here.</div></div>', unsafe_allow_html=True)
    else:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.write(f"**Loaded:** `{uploaded.name}` · **{len(uploaded_df):,} rows × {len(uploaded_df.columns)} columns")
            missing = [c for c in MODEL_INPUT_COLUMNS if c not in uploaded_df.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.dataframe(uploaded_df.head(10), use_container_width=True, hide_index=True)
                if st.button("Score uploaded dataset  →", type="primary", use_container_width=True):
                    try:
                        scored = predict_batch(bundle, uploaded_df)
                        st.session_state["batch_result"] = scored
                    except Exception as exc:
                        st.error(f"Batch inference failed: {exc}")
                scored = st.session_state.get("batch_result")
                if scored is not None:
                    st.markdown('<div class="section-title">Batch results</div>', unsafe_allow_html=True)
                    k1,k2,k3,k4 = st.columns(4)
                    approved_count = int((scored.prediction == 1).sum())
                    with k1: st.metric("Rows scored", f"{len(scored):,}")
                    with k2: st.metric("Approved", f"{approved_count:,}")
                    with k3: st.metric("Rejected", f"{len(scored)-approved_count:,}")
                    with k4:
                        if "correct" in scored.columns: st.metric("Accuracy on uploaded labels", f'{scored["correct"].mean():.2%}')
                        else: st.metric("Mean approval probability", f'{scored["approval_probability"].mean():.1%}')
                    st.dataframe(scored, use_container_width=True, hide_index=True)
                    st.download_button("Download scored CSV", data=scored.to_csv(index=False).encode("utf-8"), file_name="loan_intelligence_predictions.csv", mime="text/csv", type="primary")

# ----------------------------- Presentation -----------------------------
elif page == "Project Presentation":
    st.markdown('<div class="eyebrow mono">IN-APP EXECUTIVE + TECHNICAL DECK</div><div class="page-title">Loan Intelligence — Project Presentation</div><div class="page-sub">A complete walkthrough designed for a live NTI demo: problem → data → preprocessing → imbalance → models → decision → product → limitations → roadmap.</div>', unsafe_allow_html=True)
    slides = [
        ("01","LOAN INTELLIGENCE","A complete ML system for loan-status prediction","This project turns a binary classification problem into an end-to-end machine learning product: data exploration, preprocessing, imbalance handling, model benchmarking, reusable inference and an interactive UI.","TEAM • Yusuf Ayman Tolba · Mohamed Reda Hussein · Abdelrahman Mohamed Ahmed · Abdelmoniem Ibrahim Abdelmoniem"),
        ("02","THE PROBLEM","Can applicant data predict loan status?","The target is loan_status: a binary outcome representing whether a loan application is approved or rejected. The system learns patterns from applicant, financial, loan and credit-history variables.","Engineering goal: build a reproducible pipeline and demonstrate it through an actual inference interface — not a static notebook only."),
        ("03","DATASET","44,990 processed rows • 20 encoded features","The project uses applicant age, income, employment experience, home ownership, education, loan intent, loan amount, interest rate, loan-to-income ratio, credit history, credit score and previous defaults.","The raw repository dataset is loan_data.csv. The notebook is the research record; src/model.py is the reusable inference implementation."),
        ("04","DATA QUALITY","Clean before learning","loan_id is treated as an identifier. Selected numeric variables receive IQR clipping. Unrealistic age and employment-experience values are filtered using the project workflow. Categorical variables are encoded before modeling.","Important: correlation is not causation, and a low linear correlation does not automatically mean a feature is useless."),
        ("05","PREPROCESSING","Split → scale → balance","The data is split with random_state=42. RobustScaler is fitted on the training split, then SMOTETomek is applied to the scaled training data only. The held-out test set remains untouched by resampling.","This keeps the evaluation focused on unseen test data while addressing the training class imbalance."),
        ("06","CLASS IMBALANCE","27,991 / 8,001 before resampling","The original training distribution was strongly imbalanced between the two classes. SMOTETomek produced 27,887 samples in each class for the training stage.","Balancing changes the training distribution; therefore precision, recall and F1 remain important alongside accuracy."),
        ("07","MODEL BENCHMARK","Six classifiers were compared","Logistic Regression 86.55% · KNN 86.29% · Decision Tree 89.28% · Random Forest 89.48% · SVM 88.02% · XGBoost 92.87%.","The comparison uses the same held-out benchmark setup so the selection is evidence-based within this project."),
        ("08","THE WINNER","XGBoost / 92.87%","XGBoost achieved the strongest overall benchmark result: 92.87% test accuracy, 0.87 class-1 precision and 0.83 class-1 F1.","The selection is not universal. Logistic Regression and SVM reached 0.92 class-1 recall, higher than XGBoost's 0.80. The right model depends on the cost of errors."),
        ("09","GENERALIZATION","Why the benchmark needs context","The Decision Tree reached 100% training accuracy but 89.28% test accuracy, showing a clear generalization gap. XGBoost also has a train/test gap, but its held-out result is stronger in this benchmark.","A professional ML story reports what the model does well and where it can fail."),
        ("10","THE PRODUCT","Notebook → reusable model → interface","Loan Intelligence separates research from inference. The model module owns preprocessing and prediction; Streamlit owns the human-facing experience; tests and CI support project quality.","The UI exposes both single-applicant inference and batch CSV scoring."),
        ("11","PREDICTION STUDIO","Every raw feature is visible","The manual workspace accepts all raw applicant fields required by the inference contract. Numeric controls show dataset-derived ranges to reduce accidental invalid inputs.","The result includes approval probability, status, highlighted signals and benchmark context."),
        ("12","BATCH LAB","Real CSV in → scored CSV out","Users can download a ready-made template, upload a CSV, validate its schema, score every valid row, inspect predictions and download the results.","If loan_status is included in the upload, the app also calculates row-level correctness against the supplied labels."),
        ("13","MODEL INSIGHTS","Benchmark + feature importance","The app visualizes the six-model benchmark, the selected model's metrics and XGBoost feature importance. These are diagnostic tools, not causal explanations.","Future explainability should use methods such as SHAP and should be paired with calibration and fairness analysis."),
        ("14","LIMITATIONS","A prediction is not a lending policy","This is an educational portfolio project. The dataset may not represent a real lender population, and the system has not been audited for fairness, calibration, regulatory requirements or dataset shift.","Never use the model as the sole basis for a real financial decision."),
        ("15","PRODUCTION ROADMAP","From portfolio demo to ML system","Next steps: serialized/versioned model artifacts, FastAPI serving, SHAP explanations, calibration, subgroup fairness analysis, Docker, MLflow experiment tracking, CI/CD and drift monitoring.","The current separation between reusable model logic and UI is intentionally designed to make those upgrades possible."),
        ("16","THE TEAM","Built for the NTI Machine Learning track","Yusuf Ayman Tolba · Mohamed Reda Hussein · Abdelrahman Mohamed Ahmed · Abdelmoniem Ibrahim Abdelmoniem","Loan Intelligence • NTI Machine Learning Project • Educational demonstration"),
    ]
    if "slide" not in st.session_state: st.session_state["slide"] = 0
    idx = st.session_state["slide"]
    code,title,headline,copy,note = slides[idx]
    stat = "92.87%" if idx == 7 else ("44,990" if idx == 2 else ("27,887 × 2" if idx == 5 else ""))
    stat_html = f'<div class="big-stat">{stat}</div>' if stat else ''
    st.markdown(f'<div class="deck"><div class="slide-count">SLIDE {code} / {len(slides):02d} · {title}</div><div class="slide-title">{headline}<br><span class="slide-accent">{title}</span></div>{stat_html}<div class="slide-copy">{copy}</div><div class="quote">{note}</div><div class="deck-footer">LOAN INTELLIGENCE • NTI MACHINE LEARNING • {code}</div></div>', unsafe_allow_html=True)
    st.progress((idx+1)/len(slides))
    p,_,n = st.columns([1,3,1])
    with p:
        if st.button("← Previous", use_container_width=True, disabled=idx==0): st.session_state["slide"] -= 1; st.rerun()
    with n:
        if st.button("Next →", use_container_width=True, disabled=idx==len(slides)-1): st.session_state["slide"] += 1; st.rerun()

# ----------------------------- Model Insights -----------------------------
elif page == "Model Insights":
    st.markdown('<div class="eyebrow mono">EVALUATION / DIAGNOSTICS</div><div class="page-title">Model Insights</div><div class="page-sub">Evidence from the benchmark and the trained XGBoost model.</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    for col,value,label in [(a,f'{metrics["accuracy"]:.2%}','Test accuracy'),(b,'0.87','Class-1 precision'),(c,'0.83','Class-1 F1'),(d,'0.80','Class-1 recall')]:
        with col: st.markdown(f'<div class="card"><div class="card-k">{label}</div><div class="card-v">{value}</div><div class="card-note">Held-out benchmark result</div></div>',unsafe_allow_html=True)
    left,right=st.columns([1.05,.95],gap="large")
    with left:
        st.markdown('<div class="section-title">Model benchmark</div>',unsafe_allow_html=True)
        st.bar_chart(BENCHMARK.set_index("Model")[["Accuracy"]]/100,height=330)
    with right:
        st.markdown('<div class="section-title">Benchmark table</div>',unsafe_allow_html=True)
        display=BENCHMARK.copy(); display["Accuracy"]=display["Accuracy"].map(lambda x:f'{x:.2f}%')
        for col in ["Precision","Recall","F1"]: display[col]=display[col].map(lambda x:f'{x:.2f}')
        st.dataframe(display,use_container_width=True,hide_index=True,height=330)
    st.markdown('<div class="section-title">Top XGBoost signals</div>',unsafe_allow_html=True)
    importance=pd.DataFrame({"Feature":bundle.feature_columns,"Importance":bundle.model.feature_importances_}).sort_values("Importance",ascending=False).head(12).set_index("Feature")
    st.bar_chart(importance,height=350)
    st.caption("Feature importance is a model diagnostic, not causal evidence.")
    st.markdown('<div class="section-title">Confusion matrix</div>',unsafe_allow_html=True)
    cm=pd.DataFrame(metrics["confusion_matrix"],index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"])
    st.dataframe(cm,use_container_width=True)

# ----------------------------- Team & About -----------------------------
else:
    st.markdown('<div class="eyebrow mono">PROJECT / TEAM / BRAND</div><div class="page-title">Team & About</div><div class="page-sub">Project identity, team credits, data contract and official NTI reference.</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="card-k">Project</div><div class="card-v">LOAN INTELLIGENCE</div><div class="card-note">NTI Machine Learning track • Binary loan-status classification • Interactive decision-support demonstration</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Team</div>',unsafe_allow_html=True)
    cols=st.columns(4)
    for i,(col,name) in enumerate(zip(cols,TEAM),1):
        with col: st.markdown(f'<div class="team-card"><div class="team-number">0{i}</div><div class="team-name">{name}</div><div class="team-role">NTI ML Project Team</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Official NTI</div>',unsafe_allow_html=True)
    st.image(NTI_LOGO,width=90)
    st.markdown(f'**National Telecommunication Institute (NTI)** — the official institute website is available at {NTI_SITE}. NTI describes itself as an Egyptian center of excellence in education, applied research and technical consultation, established in 1983. urlOfficial NTI website{NTI_SITE}')
    st.markdown('<div class="section-title">Project contract</div>',unsafe_allow_html=True)
    contract=pd.DataFrame({"Layer":["Research","ML core","UI","Batch","Quality"],"Implementation":["loan.ipynb","src/model.py","app.py","CSV upload + predict_batch","tests + GitHub Actions"]})
    st.dataframe(contract,use_container_width=True,hide_index=True)
    st.markdown('<div class="risk warn"><strong>Responsible use</strong><br>This application is an educational ML demonstration. It is not a real credit-scoring or lending decision system.</div>',unsafe_allow_html=True)

st.markdown('<div class="footer-note">LOAN INTELLIGENCE • NTI MACHINE LEARNING • YUSUF AYMAN TOLBA · MOHAMED REDA HUSSEIN · ABDELRahman MOHAMED AHMED · ABDELMONIEM IBRAHIM ABDELMONIEM • EDUCATIONAL DEMONSTRATION</div>',unsafe_allow_html=True)
