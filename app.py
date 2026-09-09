from pathlib import Path
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

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#04070d;--panel:#080e17;--panel2:#0d1622;--line:#1a2a3d;--text:#f5f7fb;--muted:#8796a9;--cyan:#22d3ee;--lime:#a3e635;--purple:#a78bfa;--red:#fb7185;--amber:#f59e0b}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(4,7,13,.9)}[data-testid="stSidebar"]{background:#060b12;border-right:1px solid var(--line)}
.block-container{max-width:1500px;padding:1.2rem 2rem 4rem}#MainMenu,footer{visibility:hidden}
.mono{font-family:'DM Mono',monospace}.eyebrow{color:var(--cyan);font-size:.64rem;font-weight:800;letter-spacing:.2em;text-transform:uppercase}
.brand{padding:.2rem 0 1rem}.brand-row{display:flex;align-items:center;gap:.7rem}.brand-logo{width:40px;height:40px;border-radius:10px;background:#fff;padding:5px;object-fit:contain}.brand-mark{color:var(--cyan);font-weight:800;font-size:.75rem}.brand-title{font-size:1rem;font-weight:800}.brand-sub{font-size:.6rem;color:var(--muted);margin-top:.15rem}
.nav-label{font-size:.59rem;color:#64748b;text-transform:uppercase;letter-spacing:.18em;font-weight:800;margin:1rem 0 .45rem}.team-mini{font-size:.61rem;color:#9aa8b9;line-height:1.7;border-top:1px solid var(--line);padding-top:.75rem}.team-mini strong{color:#e2e8f0}
.hero{position:relative;overflow:hidden;padding:2.7rem;border:1px solid var(--line);border-radius:28px;background:radial-gradient(circle at 92% 8%,rgba(34,211,238,.18),transparent 26%),radial-gradient(circle at 75% 110%,rgba(167,139,250,.14),transparent 34%),linear-gradient(135deg,#08111d,#07101a 60%,#0d1020);margin-bottom:1.25rem}.hero:after{content:'';position:absolute;right:-90px;top:-150px;width:390px;height:390px;border:1px solid rgba(34,211,238,.13);border-radius:50%;box-shadow:0 0 0 35px rgba(34,211,238,.02),0 0 0 75px rgba(34,211,238,.012)}
.hero h1{margin:.4rem 0 0;font-size:clamp(2.8rem,6vw,5.7rem);line-height:.88;letter-spacing:-.075em;font-weight:800}.hero h1 span{color:var(--cyan)}.hero p{color:#a8b5c5;max-width:900px;font-size:.88rem;line-height:1.75;margin:1rem 0 0}.hero-meta{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.35rem}.pill{border:1px solid var(--line);background:rgba(255,255,255,.025);padding:.42rem .68rem;border-radius:999px;color:#abb8c7;font-size:.62rem}.pill strong{color:#fff}
.page-title{font-size:2.35rem;font-weight:800;letter-spacing:-.065em;margin:.1rem 0 .2rem}.page-sub{color:var(--muted);font-size:.78rem;margin-bottom:1.2rem}.section-title{margin:1rem 0 .6rem;font-size:.62rem;text-transform:uppercase;letter-spacing:.18em;color:#8fa0b5;font-weight:800}
.card{border:1px solid var(--line);background:linear-gradient(145deg,#09111b,#070d16);border-radius:18px;padding:1.1rem;height:100%}.card-k{color:#718198;font-size:.58rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800}.card-v{font-family:'DM Mono',monospace;font-size:1.55rem;margin-top:.3rem}.card-note{color:var(--muted);font-size:.64rem;margin-top:.3rem;line-height:1.5}
.step{display:flex;gap:.8rem;padding:.8rem 0;border-bottom:1px solid #142235}.step:last-child{border-bottom:0}.step-no{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.65rem;border:1px solid #244052;border-radius:8px;padding:.3rem .42rem}.step h4{margin:0;font-size:.76rem}.step p{margin:.15rem 0 0;color:var(--muted);font-size:.65rem;line-height:1.5}
label,.stNumberInput label,.stSelectbox label{color:#b7c1ce!important;font-size:.68rem!important;font-weight:600!important}input,[data-baseweb="select"]>div{background:#070d16!important;border-color:#1a2a3d!important;color:#f5f7fb!important;border-radius:10px!important}.stButton>button{border:1px solid rgba(34,211,238,.38);background:linear-gradient(135deg,#0c202a,#09151e);color:#e6fbff;border-radius:11px;min-height:43px;font-weight:800}.stButton>button:hover{border-color:var(--cyan);color:#fff}.stDownloadButton>button{border-radius:11px}
.range-note{font-size:.57rem;color:#66768b;margin-top:-.35rem;margin-bottom:.55rem}.decision{min-height:290px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border:1px solid var(--line);border-radius:20px;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.1),transparent 48%),var(--panel);padding:1.5rem}.decision .status{font-size:.59rem;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:800}.decision .label{margin-top:.45rem;font-size:2.65rem;line-height:1;font-weight:800;letter-spacing:-.06em}.approved{color:var(--lime)}.rejected{color:var(--red)}.prob{font-family:'DM Mono',monospace;margin-top:.8rem;font-size:1.05rem}.confidence-bar{width:82%;height:7px;background:#172232;border-radius:99px;overflow:hidden;margin-top:1rem}.confidence-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));border-radius:99px}.decision-note{color:var(--muted);font-size:.61rem;margin-top:.7rem}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:.7rem}.metric{border:1px solid var(--line);background:#090f19;border-radius:13px;padding:.8rem}.metric .k{color:var(--muted);font-size:.56rem;text-transform:uppercase;letter-spacing:.1em}.metric .v{font-family:'DM Mono',monospace;font-size:1rem;margin-top:.2rem}
.risk{padding:.65rem .75rem;border-left:2px solid var(--lime);background:rgba(163,230,53,.035);margin:.35rem 0;border-radius:0 9px 9px 0;color:#aeb9c8;font-size:.66rem;line-height:1.45}.risk.warn{border-left-color:var(--amber)}
.deck{border:1px solid var(--line);border-radius:26px;background:radial-gradient(circle at 90% 5%,rgba(34,211,238,.1),transparent 25%),linear-gradient(145deg,#0a121d,#070d16);min-height:510px;padding:2.4rem;position:relative;overflow:hidden}.deck:before{content:'LOAN\AINTELLIGENCE';white-space:pre;position:absolute;right:2rem;top:1.1rem;color:rgba(255,255,255,.035);font-size:4.5rem;font-weight:800;line-height:.8;letter-spacing:-.08em;text-align:right}.slide-count{color:var(--cyan);font-family:'DM Mono',monospace;font-size:.64rem}.slide-title{font-size:2.55rem;line-height:1.03;letter-spacing:-.065em;font-weight:800;margin:.6rem 0 .9rem;max-width:950px}.slide-copy{color:#a9b5c4;max-width:900px;line-height:1.8;font-size:.82rem}.slide-accent{color:var(--cyan)}.big-stat{font-family:'DM Mono',monospace;font-size:4rem;letter-spacing:-.08em;color:var(--cyan);margin:.8rem 0}.quote{border-left:2px solid var(--cyan);padding-left:1rem;color:#cbd5e1;font-size:.86rem;line-height:1.7;margin-top:1.2rem;max-width:900px}.deck-footer{position:absolute;left:2.4rem;right:2.4rem;bottom:1.2rem;color:#536176;font-size:.56rem;letter-spacing:.09em;text-transform:uppercase}
.team-card{border:1px solid var(--line);border-radius:16px;background:#090f18;padding:1rem}.team-number{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.62rem}.team-name{font-size:.8rem;font-weight:700;margin-top:.35rem}.team-role{color:var(--muted);font-size:.6rem;margin-top:.25rem}.footer-note{text-align:center;color:#536176;font-size:.58rem;margin-top:2rem;letter-spacing:.06em}
</style>
""",
    unsafe_allow_html=True,
)


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


def clamp_num(value, low, high):
    return max(low, min(high, value))


with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-row"><img class="brand-logo" src="{NTI_LOGO}" onerror="this.style.display=\'none\'"><div><div class="brand-mark">NTI / ML</div><div class="brand-title">Loan Intelligence</div><div class="brand-sub">End-to-end classification system</div></div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Prediction Studio", "Batch Lab", "Project Presentation", "Model Insights", "Team & About"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="nav-label">Live system</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="risk"><strong>XGBoost</strong><br>{metrics["accuracy"]:.2%} held-out test accuracy</div>'
        f'<div class="risk"><strong>{len(reference):,} source rows</strong><br>{len(bundle.feature_columns)} encoded model features</div>'
        f'<div class="risk"><strong>Training balance</strong><br>{metrics["train_rows_after_smotetomek"]:,} rows after SMOTETomek</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-label">Team</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="team-mini">' + "<br>".join(f'<strong>{i + 1}.</strong> {n}' for i, n in enumerate(TEAM)) + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="margin-top:.8rem;font-size:.6rem"><a href="{NTI_SITE}" target="_blank" style="color:#22d3ee;text-decoration:none">Official NTI website ↗</a></div>',
        unsafe_allow_html=True,
    )


if page == "Prediction Studio":
    st.markdown(
        f'<div class="hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK / FINAL PROJECT</div>'
        f'<h1>Loan <span>Intelligence.</span></h1>'
        f'<p>A real interactive binary loan-status prediction system using the same preprocessing logic and XGBoost model family benchmarked in the project notebook. Numeric controls are bounded by the supplied dataset.</p>'
        f'<div class="hero-meta"><div class="pill"><strong>{metrics["accuracy"]:.2%}</strong> test accuracy</div>'
        f'<div class="pill"><strong>{len(reference):,}</strong> source rows</div>'
        f'<div class="pill"><strong>{len(bundle.feature_columns)}</strong> encoded features</div>'
        f'<div class="pill"><strong>6</strong> classifiers benchmarked</div></div></div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.15, 0.85], gap="large")
    with form_col:
        st.markdown('<div class="section-title">01 / Applicant profile — all model inputs</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=RANGES["age"][0], max_value=RANGES["age"][1], value=int(clamp_num(30, *RANGES["age"])), step=1)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["age"][0]} → {RANGES["age"][1]}</div>', unsafe_allow_html=True)
                income = st.number_input("Annual income", min_value=RANGES["income"][0], max_value=RANGES["income"][1], value=clamp_num(60000.0, *RANGES["income"]), step=1000.0)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["income"][0]:,.0f} → {RANGES["income"][1]:,.0f}</div>', unsafe_allow_html=True)
                emp = st.number_input("Employment experience (years)", min_value=RANGES["emp_exp"][0], max_value=RANGES["emp_exp"][1], value=int(clamp_num(5, *RANGES["emp_exp"])), step=1)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["emp_exp"][0]} → {RANGES["emp_exp"][1]}</div>', unsafe_allow_html=True)
                gender = st.selectbox("Gender", ["male", "female"])
                education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2)
                home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            with c2:
                loan_amount = st.number_input("Loan amount", min_value=RANGES["loan_amount"][0], max_value=RANGES["loan_amount"][1], value=clamp_num(10000.0, *RANGES["loan_amount"]), step=500.0)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["loan_amount"][0]:,.0f} → {RANGES["loan_amount"][1]:,.0f}</div>', unsafe_allow_html=True)
                interest = st.number_input("Interest rate (%)", min_value=RANGES["interest"][0], max_value=RANGES["interest"][1], value=clamp_num(10.0, *RANGES["interest"]), step=0.1)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["interest"][0]:.2f} → {RANGES["interest"][1]:.2f}</div>', unsafe_allow_html=True)
                loan_pct = st.number_input("Loan percent of income", min_value=RANGES["loan_percent_income"][0], max_value=RANGES["loan_percent_income"][1], value=clamp_num(0.20, *RANGES["loan_percent_income"]), step=0.01, format="%.2f")
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["loan_percent_income"][0]:.2f} → {RANGES["loan_percent_income"][1]:.2f}</div>', unsafe_allow_html=True)
                credit_history = st.number_input("Credit history length", min_value=RANGES["credit_history"][0], max_value=RANGES["credit_history"][1], value=clamp_num(4.0, *RANGES["credit_history"]), step=0.5)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["credit_history"][0]:.1f} → {RANGES["credit_history"][1]:.1f}</div>', unsafe_allow_html=True)
                credit_score = st.number_input("Credit score", min_value=RANGES["credit_score"][0], max_value=RANGES["credit_score"][1], value=int(clamp_num(680, *RANGES["credit_score"])), step=1)
                st.markdown(f'<div class="range-note">Dataset range: {RANGES["credit_score"][0]} → {RANGES["credit_score"][1]}</div>', unsafe_allow_html=True)
                intent = st.selectbox("Loan intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
                previous_default = st.selectbox("Previous loan default", ["No", "Yes"])

            predict_clicked = st.button("RUN XGBOOST PREDICTION", use_container_width=True)

    with result_col:
        st.markdown('<div class="section-title">02 / Decision engine</div>', unsafe_allow_html=True)
        if predict_clicked:
            applicant = {
                "person_age": age,
                "person_income": income,
                "person_home_ownership": home,
                "person_emp_exp": emp,
                "loan_intent": intent,
                "loan_amnt": loan_amount,
                "loan_int_rate": interest,
                "loan_percent_income": loan_pct,
                "cb_person_cred_hist_length": credit_history,
                "credit_score": credit_score,
                "previous_loan_defaults_on_file": previous_default,
                "person_gender": gender,
                "person_education": education,
            }
            try:
                result = predict_one(bundle, applicant)
                label_class = "approved" if result["prediction"] == 1 else "rejected"
                st.markdown(
                    f'<div class="decision"><div class="status">XGBoost decision</div>'
                    f'<div class="label {label_class}">{result["label"]}</div>'
                    f'<div class="prob">Approval probability · {result["approval_probability"]:.1%}</div>'
                    f'<div class="confidence-bar"><div class="confidence-fill" style="width:{result["approval_probability"]:.1%}"></div></div>'
                    f'<div class="decision-note">Model output is a prediction, not a real lending decision.</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="metric-row"><div class="metric"><div class="k">Approved</div><div class="v">{result["approval_probability"]:.1%}</div></div>'
                    f'<div class="metric"><div class="k">Rejected</div><div class="v">{result["rejection_probability"]:.1%}</div></div>'
                    f'<div class="metric"><div class="k">Test accuracy</div><div class="v">{metrics["accuracy"]:.1%}</div></div></div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            st.markdown('<div class="decision"><div class="status">Ready</div><div class="label" style="color:#cbd5e1">Awaiting input</div><div class="decision-note">Enter the applicant profile and run the real trained model.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">03 / What happens under the hood</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    for col, number, title, text in [
        (a, "01", "Validate", "All required raw features are collected and bounded by the reference dataset."),
        (b, "02", "Transform", "The same encoding, IQR clipping and RobustScaler logic used during training is reused."),
        (c, "03", "Predict", "XGBoost produces a binary class and class probabilities."),
        (d, "04", "Explain", "The interface shows status, probabilities and benchmark context."),
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="card-k">{number}</div><div style="font-weight:800;margin-top:.4rem">{title}</div><div class="card-note">{text}</div></div>', unsafe_allow_html=True)


elif page == "Batch Lab":
    st.markdown('<div class="eyebrow mono">02 / OPERATIONS</div><div class="page-title">Batch Lab</div><div class="page-sub">Upload a CSV containing the raw applicant fields, score every valid row with the trained XGBoost model, inspect the results, and download the scored dataset.</div>', unsafe_allow_html=True)
    template = reference[MODEL_INPUT_COLUMNS].head(3).copy()
    template.insert(0, "loan_id", range(1, len(template) + 1))
    buf = BytesIO()
    template.to_csv(buf, index=False)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("DOWNLOAD CSV TEMPLATE", data=buf.getvalue(), file_name="loan_prediction_template.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.metric("Required raw fields", len(MODEL_INPUT_COLUMNS))
    with c3:
        st.metric("Model features", len(bundle.feature_columns))
    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [c for c in MODEL_INPUT_COLUMNS if c not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.success(f"Schema valid · {len(batch):,} rows loaded")
                with st.expander("Preview uploaded data", expanded=False):
                    st.dataframe(batch.head(10), use_container_width=True)
                if st.button("SCORE ENTIRE FILE", use_container_width=True):
                    try:
                        scored = predict_batch(bundle, batch)
                        approved = int((scored["prediction"] == 1).sum())
                        rejected = int((scored["prediction"] == 0).sum())
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Approved", f"{approved:,}")
                        m2.metric("Rejected", f"{rejected:,}")
                        m3.metric("Scored rows", f"{len(scored):,}")
                        if "actual_status" in scored.columns:
                            batch_acc = scored["correct"].mean()
                            st.metric("Accuracy vs provided loan_status", f"{batch_acc:.2%}")
                        st.dataframe(scored, use_container_width=True, height=420)
                        out = BytesIO()
                        scored.to_csv(out, index=False)
                        st.download_button("DOWNLOAD SCORED CSV", data=out.getvalue(), file_name="loan_predictions_scored.csv", mime="text/csv", use_container_width=True)
                    except Exception as exc:
                        st.error(f"Batch prediction failed: {exc}")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")


elif page == "Project Presentation":
    st.markdown('<div class="eyebrow mono">03 / PROJECT STORY</div><div class="page-title">Project Presentation</div><div class="page-sub">A presentation built inside the product: problem → data → preprocessing → imbalance → benchmark → model selection → product → limitations → roadmap.</div>', unsafe_allow_html=True)
    slides = [
        ("01", "Loan Intelligence", "An end-to-end machine learning classification system for predicting observed loan status from borrower, loan and credit information.", "NTI • Machine Learning Track • Final Project", "92.87%", "XGBoost test accuracy"),
        ("02", "The problem", "Loan-status prediction combines many interacting applicant and loan attributes. A useful ML system should make the workflow consistent, measurable and repeatable rather than relying on ad-hoc rules.", "Project question: can a classification pipeline predict loan status accurately while handling data quality and class imbalance?", None, None),
        ("03", "Project objectives", "Understand the data, clean and transform mixed features, handle class imbalance, benchmark six classification strategies, select a strong model on held-out data, and expose the final model through an interactive UI.", "The deliverable is deliberately more than a notebook: it is a reproducible ML workflow plus a usable prediction interface.", None, None),
        ("04", "Dataset snapshot", f"The project works with {len(reference):,} source rows and the raw applicant fields represented in the CSV. After preprocessing and encoding, the modeling matrix contains {len(bundle.feature_columns)} features.", "Target: loan_status. The modeling split is 80/20 with random_state=42.", f"{len(reference):,}", "source rows"),
        ("05", "What the model sees", "Borrower profile: age, income, employment experience, education and gender. Loan profile: amount, interest rate, loan-to-income ratio and intent. Credit context: credit score, credit-history length and previous defaults. Ownership is categorical.", "The raw categorical values are converted into model-ready numeric representations.", str(len(MODEL_INPUT_COLUMNS)), "raw model inputs"),
        ("06", "EDA & key signals", "Exploration covered distributions, missing values, duplicates, categorical counts, boxplots, scatterplots, target distribution and correlations. The strongest observed target relationships included previous defaults, loan-to-income ratio, interest rate, rent ownership and income.", "Correlation is a screening signal, not proof of causation. A near-zero linear correlation does not prove a feature is useless.", "0.54", "strongest observed correlation"),
        ("07", "Data quality & cleaning", "loan_id is removed because it is an identifier rather than a predictive attribute. IQR clipping is applied to income, loan amount, interest rate, loan-percent-income, credit-history length and credit score. Age is constrained to ≤80 and employment experience to ≤60.", "These operations match the project notebook pipeline used for the benchmark.", None, None),
        ("08", "Encoding mixed data", "Gender and previous-default status are mapped to binary values. Education is encoded from High School through Doctorate as 1→5. Home ownership and loan intent are one-hot encoded with drop_first=True.", "The education mapping is ordinal; one-hot encoding would be a neutral alternative if the project were hardened further.", None, None),
        ("09", "Robust scaling", "RobustScaler uses the median and interquartile range, making it less sensitive to extreme values than mean/std scaling. The scaler is fitted on the training split and then applied to the test split.", "This keeps the model's numeric representation consistent between training and inference.", None, None),
        ("10", "Class imbalance", "Before resampling, the training data contained 27,991 class-0 rows and 8,001 class-1 rows. SMOTETomek produced a balanced training set of 27,887 rows per class.", "SMOTE creates minority examples while Tomek-link cleaning removes ambiguous overlaps from the resampled training distribution.", "27,887", "rows per class after SMOTETomek"),
        ("11", "Six-model benchmark", "The project compares Logistic Regression, KNN, Decision Tree, Random Forest, SVM and XGBoost on the same held-out test set. This prevents selecting an algorithm simply because it is familiar.", "Benchmark accuracy: Logistic 86.55% • KNN 86.29% • Tree 89.28% • Random Forest 89.48% • SVM 88.02% • XGBoost 92.87%.", "6", "classifiers compared"),
        ("12", "Why XGBoost?", "XGBoost delivered the highest test accuracy in this benchmark and the strongest class-1 precision/F1 combination among the tested models. It is therefore the strongest overall choice for this project benchmark—not a universal winner.", "XGBoost: 92.87% accuracy • 0.87 class-1 precision • 0.80 class-1 recall • 0.83 class-1 F1.", "92.87%", "held-out test accuracy"),
        ("13", "Accuracy is not the whole story", "Logistic Regression and SVM reached 0.92 class-1 recall, higher than XGBoost's 0.80. If missing a positive case were much more expensive than producing a false positive, the preferred operating point could change.", "Model choice depends on the business cost of errors, not only on the largest accuracy number.", "0.80", "XGBoost class-1 recall"),
        ("14", "Generalization & overfitting", "The Decision Tree reached 100% training accuracy but only 89.28% test accuracy, a clear overfitting warning. XGBoost also has a train/test gap (97.68% vs 92.87%), so the final model should not be presented as perfect.", "The test result is evidence from this split—not proof of production performance on future populations.", "4.81 pts", "XGBoost train/test gap"),
        ("15", "From notebook to product", "The application reuses the same transformation path for inference, then feeds the processed row to the trained XGBoost classifier. The UI exposes single-row prediction, batch scoring, probabilities, benchmark context and model insights.", "The product layer makes the ML pipeline demonstrable and repeatable instead of leaving the result inside a notebook.", None, None),
        ("16", "Prediction Studio", "Users enter all 13 raw model inputs. Numeric fields are bounded by the supplied dataset. The application runs the real trained model and displays Approved/Rejected plus approval and rejection probabilities.", "Same raw schema → same preprocessing logic → XGBoost inference → decision output.", "13", "raw input fields"),
        ("17", "Batch Lab", "A CSV template can be downloaded, populated with multiple applicants and uploaded back into the app. The system validates the required schema, scores every valid row, optionally compares predictions with a supplied loan_status column, and exports the scored CSV.", "This is the bridge from a single demo prediction to repeatable data processing.", None, None),
        ("18", "Model insights", "The app exposes the six-model benchmark, XGBoost feature importance, the confusion matrix and the precision/recall trade-off. These views connect the numerical benchmark to a model-selection argument.", "Correlation and model feature importance answer different questions and should not be conflated.", None, None),
        ("19", "Responsible interpretation", "The project does not claim that 92.87% accuracy means every future applicant will be classified correctly. It also does not claim causality from correlations. Dataset representativeness, fairness, calibration, monitoring and governance require deeper validation before real lending use.", "A strong ML defense is honest about scope and uncertainty.", None, None),
        ("20", "Production roadmap", "Next steps: cross-validation, hyperparameter tuning, threshold/cost analysis, SHAP explanations, fairness audits, drift monitoring, model/version tracking, stronger input validation, secure deployment and a dedicated API layer.", "The current application is a project prototype and demonstration—not a production credit-risk system.", None, None),
        ("21", "The complete ML lifecycle", "DATA → CLEANING → ENCODING → SCALING → BALANCING → BENCHMARKING → XGBOOST → INFERENCE → PRODUCT.", "The value of the project is the complete, reproducible path from raw data to an interactive decision interface.", "DATA → MODEL → PRODUCT", "end-to-end workflow"),
        ("22", "Team & closing", "Yusuf Ayman Tolba • Mohamed Reda Hussein • Abdelrahman Mohamed Ahmed • Abdelmoniem Ibrahim Abdelmoniem", "Built for the NTI Machine Learning track. Thank you.", "Q&A", "Let's run the model."),
    ]
    if "slide_idx" not in st.session_state:
        st.session_state.slide_idx = 0
    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("← PREVIOUS", use_container_width=True):
            st.session_state.slide_idx = max(0, st.session_state.slide_idx - 1)
    with nav2:
        chosen = st.slider("Slide", 1, len(slides), st.session_state.slide_idx + 1, label_visibility="collapsed")
        st.session_state.slide_idx = chosen - 1
    with nav3:
        if st.button("NEXT →", use_container_width=True):
            st.session_state.slide_idx = min(len(slides) - 1, st.session_state.slide_idx + 1)
    num, title, body, note, stat, stat_label = slides[st.session_state.slide_idx]
    stat_html = f'<div class="big-stat">{stat}</div><div class="slide-copy">{stat_label}</div>' if stat else ''
    st.markdown(
        f'<div class="deck"><div class="slide-count">SLIDE {num} / {len(slides):02d}</div><div class="slide-title">{title}</div>'
        f'<div class="slide-copy">{body}</div>{stat_html}<div class="quote">{note}</div>'
        f'<div class="deck-footer">NTI • MACHINE LEARNING • LOAN INTELLIGENCE • PROJECT PRESENTATION</div></div>',
        unsafe_allow_html=True,
    )


elif page == "Model Insights":
    st.markdown('<div class="eyebrow mono">04 / EVIDENCE</div><div class="page-title">Model Insights</div><div class="page-sub">The benchmark, final-model metrics, confusion matrix and model-level feature importance behind the live application.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f'{metrics["accuracy"]:.2%}')
    c2.metric("Class 1 precision", f'{metrics["classification_report"]["1"]["precision"]:.2f}')
    c3.metric("Class 1 recall", f'{metrics["classification_report"]["1"]["recall"]:.2f}')
    c4.metric("Class 1 F1", f'{metrics["classification_report"]["1"]["f1-score"]:.2f}')

    st.markdown('<div class="section-title">Benchmark comparison</div>', unsafe_allow_html=True)
    display_benchmark = BENCHMARK.copy()
    for col in ["Precision", "Recall", "F1"]:
        display_benchmark[col] = display_benchmark[col].map(lambda x: f"{x:.2f}")
    display_benchmark["Accuracy"] = display_benchmark["Accuracy"].map(lambda x: f"{x:.2f}%")
    st.dataframe(display_benchmark, use_container_width=True, hide_index=True)
    st.bar_chart(BENCHMARK.set_index("Model")["Accuracy"], use_container_width=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="section-title">XGBoost feature importance</div>', unsafe_allow_html=True)
        importance = pd.DataFrame({"Feature": bundle.feature_columns, "Importance": bundle.model.feature_importances_}).sort_values("Importance", ascending=False).head(12).set_index("Feature")
        st.bar_chart(importance, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Confusion matrix — held-out test set</div>', unsafe_allow_html=True)
        cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        st.dataframe(cm, use_container_width=True)
        st.markdown('<div class="risk warn"><strong>Interpretation:</strong> XGBoost class-1 recall is 0.80, so some positive cases are missed. Accuracy alone should not be treated as the full evaluation.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model selection argument</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-k">Decision</div><div style="font-size:1.35rem;font-weight:800;margin-top:.35rem">XGBoost is the strongest overall model in this benchmark.</div><div class="card-note">It has the highest held-out accuracy (92.87%) and strong class-1 precision/F1. Logistic Regression and SVM have higher class-1 recall (0.92), so the final choice is a trade-off rather than a universal claim.</div></div>', unsafe_allow_html=True)


elif page == "Team & About":
    st.markdown('<div class="eyebrow mono">05 / IDENTITY</div><div class="page-title">Team & About</div><div class="page-sub">Project identity, team members, technical scope and responsible-use framing.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (col, name) in enumerate(zip(cols, TEAM), start=1):
        with col:
            st.markdown(f'<div class="team-card"><div class="team-number">MEMBER {i:02d}</div><div class="team-name">{name}</div><div class="team-role">NTI Machine Learning Project</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project identity</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="card"><div class="card-k">Track</div><div class="card-v">ML</div><div class="card-note">Machine Learning — binary classification</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-k">Final model</div><div class="card-v">XGB</div><div class="card-note">XGBoost classifier</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="card"><div class="card-k">Benchmark</div><div class="card-v">{metrics["accuracy"]:.2%}</div><div class="card-note">Held-out test accuracy</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Responsible use</div>', unsafe_allow_html=True)
    st.info("This project is an educational ML prototype. It predicts patterns in the supplied dataset and should not be used as an actual lending decision system without deeper validation, fairness analysis, calibration, security, monitoring and governance.")
    st.markdown(f'<div class="footer-note">NTI • Loan Intelligence • Built by the project team • <a href="{NTI_SITE}" target="_blank" style="color:#22d3ee">National Telecommunication Institute</a></div>', unsafe_allow_html=True)
