from pathlib import Path
import html

import pandas as pd
import streamlit as st

from src.model import MODEL_INPUT_COLUMNS, predict_batch, predict_one, train_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "loan_data.csv"

st.set_page_config(
    page_title="Loan Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design system
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#02050a;--panel:#07111c;--panel2:#091522;--line:#17344d;--text:#f6fbff;--muted:#8194aa;--cyan:#22d3ee;--blue:#3b82f6;--purple:#a78bfa;--lime:#a3e635;--red:#fb7185;--amber:#fbbf24}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:radial-gradient(circle at 80% 0%,rgba(34,211,238,.07),transparent 28%),radial-gradient(circle at 10% 100%,rgba(167,139,250,.05),transparent 25%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(2,5,10,.86)}[data-testid="stSidebar"]{background:#030911;border-right:1px solid var(--line)}
.block-container{max-width:1580px;padding:1rem 2rem 4rem}#MainMenu,footer{visibility:hidden}
.stButton>button{border:1px solid #21415a;background:linear-gradient(135deg,#081723,#06111a);color:#e9fbff;border-radius:12px;min-height:42px;font-weight:800;transition:.2s}.stButton>button:hover{border-color:var(--cyan);transform:translateY(-1px);box-shadow:0 10px 32px rgba(34,211,238,.12)}
.stDownloadButton>button{border-radius:12px}.stNumberInput input,[data-baseweb="select"]>div{background:#06101a!important;border-color:#17344d!important;color:#f7fbff!important;border-radius:10px!important}.stSlider label,.stNumberInput label,.stSelectbox label,.stFileUploader label{font-size:.67rem!important;color:#aebdcd!important}

/* Shared cards */
.kicker{font:500 .62rem 'DM Mono',monospace;color:var(--cyan);letter-spacing:.2em;text-transform:uppercase}.mono{font-family:'DM Mono',monospace}.muted{color:var(--muted)}
.card{height:100%;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,rgba(9,21,34,.94),rgba(5,11,18,.96));padding:1.05rem;box-shadow:0 16px 45px rgba(0,0,0,.18);transition:.25s}.card:hover{border-color:#24516d;transform:translateY(-2px);box-shadow:0 18px 55px rgba(0,0,0,.25)}
.card-k{font-size:.55rem;color:#71869d;text-transform:uppercase;letter-spacing:.14em;font-weight:800}.card-v{font:500 1.45rem 'DM Mono',monospace;margin-top:.28rem;color:#fff}.card-note{font-size:.61rem;color:var(--muted);line-height:1.55;margin-top:.3rem}
.section{margin:1.35rem 0 .7rem}.section-title{font-size:.62rem;color:#8297ad;text-transform:uppercase;letter-spacing:.18em;font-weight:800}.section-line{height:1px;background:linear-gradient(90deg,var(--line),transparent);margin-top:.4rem}

/* Hero */
.hero{position:relative;overflow:hidden;min-height:350px;border:1px solid #1b4662;border-radius:30px;padding:2.8rem;background:radial-gradient(circle at 78% 28%,rgba(34,211,238,.2),transparent 20%),radial-gradient(circle at 91% 5%,rgba(167,139,250,.16),transparent 28%),linear-gradient(135deg,#061321,#07101c 55%,#0b0c20);box-shadow:0 30px 110px rgba(0,0,0,.34)}
.hero:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(90deg,transparent,black 20%,black 78%,transparent);pointer-events:none}.hero:after{content:'';position:absolute;right:-120px;top:-170px;width:500px;height:500px;border:1px solid rgba(34,211,238,.12);border-radius:50%;box-shadow:0 0 0 38px rgba(34,211,238,.025),0 0 0 78px rgba(167,139,250,.018),0 0 100px rgba(34,211,238,.09);animation:orbit 7s ease-in-out infinite}@keyframes orbit{50%{transform:translate(-12px,12px) scale(1.02)}}
.hero-content{position:relative;z-index:2;max-width:900px}.hero h1{font-size:clamp(3.4rem,7vw,7rem);line-height:.84;letter-spacing:-.09em;margin:.5rem 0 1rem;font-weight:800}.hero h1 span{color:var(--cyan)}.hero-sub{font-size:.92rem;line-height:1.8;color:#b8c7d6;max-width:800px}.hero-pills{display:flex;gap:.45rem;flex-wrap:wrap;margin-top:1.25rem}.pill{border:1px solid #1e4059;background:rgba(255,255,255,.035);border-radius:99px;padding:.45rem .7rem;font:500 .56rem 'DM Mono';color:#b9c8d6}.pill b{color:#fff}.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--lime);box-shadow:0 0 12px var(--lime);margin-right:5px;animation:pulse 1.4s infinite}@keyframes pulse{50%{opacity:.35}}

/* Visual primitives */
.glow-number{font:500 clamp(2.5rem,5vw,5rem) 'DM Mono';letter-spacing:-.08em;color:var(--cyan);line-height:.9}.label{font-size:.57rem;color:#71869d;text-transform:uppercase;letter-spacing:.15em;font-weight:800}.chips{margin-top:.8rem}.chip{display:inline-block;border:1px solid #1c3a53;background:#07131f;border-radius:99px;padding:.35rem .55rem;margin:.16rem;color:#b9c7d5;font:500 .53rem 'DM Mono'}.chip b{color:var(--cyan)}
.flow{display:grid;grid-template-columns:repeat(7,1fr);gap:.45rem;align-items:stretch}.flow-item{position:relative;border:1px solid var(--line);border-radius:14px;background:#07131e;padding:.85rem .45rem;text-align:center;transition:.2s}.flow-item:hover{border-color:var(--cyan);transform:translateY(-3px)}.flow-item strong{display:block;font-size:.62rem}.flow-item small{display:block;color:var(--muted);font-size:.5rem;margin-top:.3rem;line-height:1.35}.flow-arrow{display:flex;align-items:center;justify-content:center;color:var(--cyan);font-size:1rem}
.bar-row{display:grid;grid-template-columns:170px 1fr 62px;gap:9px;align-items:center;margin:.62rem 0}.bar-name{font-size:.6rem;color:#b9c7d5}.bar-track{height:10px;background:#101e2d;border:1px solid #183047;border-radius:99px;overflow:hidden}.bar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--cyan));transform-origin:left;animation:grow .8s ease both}.bar-fill.win{background:linear-gradient(90deg,var(--cyan),var(--lime))}@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}.bar-value{font:500 .59rem 'DM Mono';text-align:right;color:#e8f7fb}
.cm{display:grid;grid-template-columns:95px 1fr 1fr;gap:6px;max-width:580px}.cm>div{padding:1rem .5rem;border:1px solid var(--line);border-radius:12px;background:#07131e;text-align:center}.cm .head{font-size:.5rem;color:#70849a;background:transparent;border:0}.cm b{font:500 1.35rem 'DM Mono'}.tn b{color:var(--cyan)}.tp b{color:var(--lime)}.fp b{color:var(--red)}.fn b{color:var(--amber)}.cm small{display:block;color:var(--muted);font-size:.5rem;margin-top:.2rem}
.callout{border-left:2px solid var(--cyan);background:rgba(34,211,238,.035);border-radius:0 13px 13px 0;padding:.8rem 1rem;color:#b9c8d6;font-size:.66rem;line-height:1.65}.callout.warn{border-color:var(--amber)}.callout.good{border-color:var(--lime)}.callout.bad{border-color:var(--red)}

/* Presentation */
.deck{position:relative;overflow:hidden;min-height:610px;border:1px solid #1b4662;border-radius:30px;padding:3rem;background:radial-gradient(circle at 87% 8%,rgba(34,211,238,.14),transparent 24%),radial-gradient(circle at 12% 100%,rgba(167,139,250,.1),transparent 30%),linear-gradient(145deg,#061321,#040a11 70%,#0a0d1c);box-shadow:0 30px 100px rgba(0,0,0,.32);animation:slideIn .45s ease both}@keyframes slideIn{from{opacity:0;transform:translateY(14px) scale(.99)}to{opacity:1;transform:none}}.deck-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:38px 38px;mask-image:linear-gradient(to bottom,black,transparent);pointer-events:none}.deck-content{position:relative;z-index:2}.deck-title{font-size:clamp(2.7rem,6vw,6rem);line-height:.88;letter-spacing:-.085em;font-weight:800;margin:.7rem 0 1rem;max-width:1100px}.deck-title span{color:var(--cyan)}.deck-sub{font-size:1rem;color:#d7e3ed;line-height:1.6;max-width:900px}.deck-body{font-size:.74rem;color:#9aacbf;line-height:1.8;max-width:930px;margin-top:1rem}.deck-stat{font:500 clamp(3rem,7vw,7rem) 'DM Mono';letter-spacing:-.09em;color:var(--lime);line-height:.9;margin:1.4rem 0 .35rem}.deck-footer{position:absolute;bottom:1.25rem;left:3rem;right:3rem;display:flex;align-items:center;gap:.8rem;z-index:3}.deck-progress{height:3px;flex:1;background:#132334;border-radius:99px;overflow:hidden}.deck-progress>div{height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple),var(--lime));transition:.4s}.deck-counter{font:500 .57rem 'DM Mono';color:#71849a}.kbd{font:500 .53rem 'DM Mono';border:1px solid #29445d;background:#07131f;padding:.2rem .35rem;border-radius:5px;color:#aebfd0}

/* Sidebar */
.brand{padding:.4rem 0 1rem}.brand-title{font-size:1rem;font-weight:800}.brand-title span{color:var(--cyan)}.brand-sub{font-size:.57rem;color:var(--muted);margin-top:.12rem}.side-status{border:1px solid #183b52;border-radius:13px;background:#07131e;padding:.7rem;margin-top:1rem;font-size:.57rem;color:#aebdcd;line-height:1.55}.side-status b{color:#fff}.side-links{font-size:.57rem;color:#708399;line-height:2;margin-top:1rem}.side-links b{color:#b8c8d7}
.footer-note{text-align:center;color:#4f647b;font:500 .53rem 'DM Mono';letter-spacing:.08em;margin:2rem 0 .5rem}
@media(max-width:900px){.flow{grid-template-columns:1fr 1fr}.hero{padding:1.7rem}.deck{padding:2rem;min-height:560px}.deck-footer{left:2rem;right:2rem}.bar-row{grid-template-columns:120px 1fr 55px}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data/model loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("loan_data.csv was not found in the project root.")
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training XGBoost model…")
def load_model():
    return train_model(DATA_PATH)


try:
    df = load_data()
    bundle, metrics = load_model()
except Exception as exc:
    st.error(f"Project startup failed: {exc}")
    st.stop()

BENCHMARK = pd.DataFrame(
    [
        ["Logistic Regression", 86.55, 64, 92, 75],
        ["KNN", 86.29, 64, 88, 74],
        ["Decision Tree", 89.28, 73, 82, 77],
        ["Random Forest", 89.48, 71, 88, 79],
        ["SVM", 88.02, 67, 92, 77],
        ["XGBoost", 92.87, 87, 80, 83],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"],
)


def esc(value) -> str:
    return html.escape(str(value))


def card(title, value, note="") -> str:
    return (
        f'<div class="card"><div class="card-k">{esc(title)}</div>'
        f'<div class="card-v">{esc(value)}</div><div class="card-note">{esc(note)}</div></div>'
    )


def section(title: str) -> None:
    st.markdown(
        f'<div class="section"><div class="section-title">{esc(title)}</div><div class="section-line"></div></div>',
        unsafe_allow_html=True,
    )


def chips(items: list[str]) -> str:
    return '<div class="chips">' + "".join(f'<span class="chip">{esc(x)}</span>' for x in items) + "</div>"


def benchmark_bars() -> str:
    parts = []
    for _, row in BENCHMARK.iterrows():
        winner = " win" if row["Model"] == "XGBoost" else ""
        parts.append(
            f'<div class="bar-row"><div class="bar-name">{esc(row["Model"])}</div>'
            f'<div class="bar-track"><div class="bar-fill{winner}" style="width:{row["Accuracy"]}%"></div></div>'
            f'<div class="bar-value">{row["Accuracy"]:.2f}%</div></div>'
        )
    return "".join(parts)


def confusion_matrix_html() -> str:
    tn, fp, fn, tp = metrics["confusion_matrix"][0] + metrics["confusion_matrix"][1]
    return f'''
    <div class="cm">
      <div class="head"></div><div class="head">PREDICTED 0</div><div class="head">PREDICTED 1</div>
      <div class="head">ACTUAL 0</div><div class="tn"><b>{tn}</b><small>True Negative</small></div><div class="fp"><b>{fp}</b><small>False Positive</small></div>
      <div class="head">ACTUAL 1</div><div class="fn"><b>{fn}</b><small>False Negative</small></div><div class="tp"><b>{tp}</b><small>True Positive</small></div>
    </div>'''


def feature_bars() -> str:
    names = list(bundle.feature_columns)
    values = list(bundle.model.feature_importances_)
    ranked = sorted(zip(names, values), key=lambda x: x[1], reverse=True)[:8]
    max_value = max(values) if values else 1
    return "".join(
        f'<div class="bar-row"><div class="bar-name">{esc(name)}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{value/max_value*100:.1f}%"></div></div>'
        f'<div class="bar-value">{value:.3f}</div></div>'
        for name, value in ranked
    )


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------

def render_overview() -> None:
    accuracy = metrics["accuracy"] * 100
    st.markdown(
        f'''
        <div class="hero">
          <div class="hero-content">
            <div class="kicker"><span class="live-dot"></span>NTI MACHINE LEARNING TRACK · LIVE MODEL</div>
            <h1>LOAN <span>INTELLIGENCE</span></h1>
            <div class="hero-sub">AI-powered loan-status classification presented as an interactive product: real data, reusable preprocessing, class-imbalance handling, model benchmarking, XGBoost inference, and honest evaluation.</div>
            <div class="hero-pills">
              <span class="pill"><b>{len(df):,}</b> source rows</span>
              <span class="pill"><b>{len(MODEL_INPUT_COLUMNS)}</b> raw inputs</span>
              <span class="pill"><b>{accuracy:.2f}%</b> test accuracy</span>
              <span class="pill"><b>XGBoost</b> selected model</span>
            </div>
          </div>
        </div>''',
        unsafe_allow_html=True,
    )

    section("Project pulse")
    cols = st.columns(4)
    values = [
        ("Dataset", f"{len(df):,}", "rows loaded from loan_data.csv"),
        ("Encoded space", len(bundle.feature_columns), "features after preprocessing"),
        ("Test accuracy", f"{accuracy:.2f}%", "held-out evaluation"),
        ("Balanced train", f"{metrics['train_rows_after_smotetomek']:,}", "after SMOTETomek"),
    ]
    for col, (title, value, note) in zip(cols, values):
        with col:
            st.markdown(card(title, value, note), unsafe_allow_html=True)

    section("End-to-end pipeline")
    flow = [
        ("01", "Raw Data", "CSV"),
        ("02", "Cleaning", "IQR + rules"),
        ("03", "Encoding", "categorical"),
        ("04", "Scaling", "RobustScaler"),
        ("05", "Balance", "SMOTETomek"),
        ("06", "Model", "XGBoost"),
        ("07", "Decision", "0 / 1"),
    ]
    flow_html = '<div class="flow">' + "".join(
        f'<div class="flow-item"><strong>{a} · {b}</strong><small>{c}</small></div>' for a, b, c in flow
    ) + '</div>'
    st.markdown(flow_html, unsafe_allow_html=True)

    section("Model arena")
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown(card("Benchmark", "6 models", "same project evaluation contract"), unsafe_allow_html=True)
        st.markdown(benchmark_bars(), unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="card"><div class="card-k">Selected model</div><div class="glow-number">XGBoost</div>'
            '<div class="card-note">Highest test accuracy in the documented benchmark, with class-1 precision 0.87 and F1 0.83.</div>'
            + chips(["92.87% ACCURACY", "0.87 PRECISION", "0.80 RECALL", "0.83 F1"])
            + '</div>',
            unsafe_allow_html=True,
        )

    section("Evaluation at a glance")
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.markdown(confusion_matrix_html(), unsafe_allow_html=True)
        st.markdown('<div class="callout good" style="margin-top:.8rem">Held-out test data remains separate from SMOTETomek. The model is evaluated on data it did not see during balancing.</div>', unsafe_allow_html=True)
    with c2:
        report = metrics["classification_report"]
        rows = []
        for label in ["0", "1"]:
            rows.append({"Class": label, "Precision": report[label]["precision"], "Recall": report[label]["recall"], "F1": report[label]["f1-score"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_data() -> None:
    st.markdown('<div class="kicker">02 / DATA & FEATURES</div><div class="deck-title">Meet the <span>data.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">The model combines demographic, employment, loan, credit, ownership, intent, default-history, gender, and education signals.</div>', unsafe_allow_html=True)

    section("Schema snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(card("Rows", f"{len(df):,}", "loaded source records"), unsafe_allow_html=True)
    with c2:
        st.markdown(card("Raw inputs", len(MODEL_INPUT_COLUMNS), "features sent to the model contract"), unsafe_allow_html=True)
    with c3:
        class_counts = df["loan_status"].value_counts().to_dict()
        st.markdown(card("Target", "loan_status", f"class 0: {class_counts.get(0, 0):,} · class 1: {class_counts.get(1, 0):,}"), unsafe_allow_html=True)

    section("Feature explorer")
    st.dataframe(df[MODEL_INPUT_COLUMNS + ["loan_status"]].head(20), use_container_width=True, hide_index=True)
    section("Target distribution")
    target = df["loan_status"].value_counts().sort_index().rename(index={0: "Rejected / 0", 1: "Approved / 1"})
    st.bar_chart(target)


def render_preprocessing() -> None:
    st.markdown('<div class="kicker">03 / PREPROCESSING</div><div class="deck-title">Make the signal <span>model-ready.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">The preprocessing contract is explicit, reusable, and shared by training and inference.</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Identifier", "Remove loan_id from predictive features."),
        ("02", "Outliers", "IQR clipping on selected numeric variables."),
        ("03", "Sanity", "Age ≤ 80 and employment experience ≤ 60."),
        ("04", "Encoding", "Binary, ordinal, and one-hot mappings."),
        ("05", "Scaling", "RobustScaler fitted on training data."),
        ("06", "Balance", "SMOTETomek applied to training only."),
    ]
    for i in range(0, len(steps), 3):
        cols = st.columns(3)
        for col, (num, title, body) in zip(cols, steps[i:i+3]):
            with col:
                st.markdown(f'<div class="card"><div class="card-k">{num}</div><div style="font-size:1rem;font-weight:800;margin-top:.35rem">{esc(title)}</div><div class="card-note" style="margin-top:.55rem">{esc(body)}</div></div>', unsafe_allow_html=True)

    section("Why RobustScaler?")
    st.markdown('<div class="callout">RobustScaler uses statistics that are less sensitive to extreme values, which fits a dataset where income, loan amount, and credit-related variables can have very different magnitudes.</div>', unsafe_allow_html=True)
    section("Encoded feature space")
    st.markdown(chips(bundle.feature_columns), unsafe_allow_html=True)


def render_evaluation() -> None:
    st.markdown('<div class="kicker">04 / EVALUATION</div><div class="deck-title">Measure what <span>matters.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">Accuracy is only one view. Precision, recall, F1, and the confusion matrix expose different failure patterns.</div>', unsafe_allow_html=True)
    report = metrics["classification_report"]["1"]
    cols = st.columns(4)
    for col, title, value, note in [
        (cols[0], "Accuracy", f"{metrics['accuracy']:.2%}", "overall correctness"),
        (cols[1], "Precision", f"{report['precision']:.2%}", "class-1 predicted positives"),
        (cols[2], "Recall", f"{report['recall']:.2%}", "class-1 cases recovered"),
        (cols[3], "F1", f"{report['f1-score']:.2%}", "precision / recall balance"),
    ]:
        with col:
            st.markdown(card(title, value, note), unsafe_allow_html=True)

    section("Confusion matrix")
    a, b = st.columns([1, 1.1])
    with a:
        st.markdown(confusion_matrix_html(), unsafe_allow_html=True)
    with b:
        st.markdown('<div class="callout warn"><b>Read the errors:</b> false positives and false negatives have different meanings. In a real lending workflow, their business cost should influence threshold and policy decisions.</div>', unsafe_allow_html=True)

    section("Model comparison")
    st.dataframe(BENCHMARK, use_container_width=True, hide_index=True)
    st.markdown(benchmark_bars(), unsafe_allow_html=True)


def render_prediction() -> None:
    st.markdown('<div class="kicker">05 / LIVE MODEL</div><div class="deck-title">Prediction <span>Studio.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">Change an applicant profile and run the exact reusable XGBoost inference function from <span class="mono">src/model.py</span>.</div>', unsafe_allow_html=True)

    home_values = sorted(df["person_home_ownership"].dropna().astype(str).unique())
    intent_values = sorted(df["loan_intent"].dropna().astype(str).unique())
    gender_values = sorted(df["person_gender"].dropna().astype(str).unique())
    default_values = sorted(df["previous_loan_defaults_on_file"].dropna().astype(str).unique())
    education_values = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]

    left, right = st.columns([1, 1])
    with left:
        age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
        income = st.number_input("Annual income", min_value=0.0, value=50000.0, step=1000.0)
        emp_exp = st.number_input("Employment experience (years)", min_value=0, max_value=60, value=5, step=1)
        home = st.selectbox("Home ownership", home_values)
        intent = st.selectbox("Loan intent", intent_values)
        loan_amount = st.number_input("Loan amount", min_value=0.0, value=10000.0, step=500.0)
    with right:
        interest = st.number_input("Interest rate (%)", min_value=0.0, value=10.0, step=0.1)
        loan_percent_income = st.number_input("Loan percent of income", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        credit_history = st.number_input("Credit history length", min_value=0.0, value=5.0, step=0.5)
        credit_score = st.number_input("Credit score", min_value=0, max_value=1000, value=650, step=1)
        previous_default = st.selectbox("Previous loan defaults", default_values)
        gender = st.selectbox("Gender", gender_values)
        education = st.selectbox("Education", education_values)

    if st.button("RUN XGBOOST PREDICTION", type="primary", use_container_width=True):
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
        approved = result["prediction"] == 1
        accent = "approved" if approved else "rejected"
        st.markdown(
            f'''<div class="card" style="margin-top:1rem;text-align:center;padding:2rem;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.12),transparent 50%),#06101a">
            <div class="card-k">LIVE MODEL DECISION</div><div class="{accent}" style="font-size:3.4rem;font-weight:800;letter-spacing:-.07em;color:{'var(--lime)' if approved else 'var(--red)'};margin-top:.35rem">{esc(result['label'])}</div>
            <div class="glow-number" style="font-size:2.6rem;margin-top:.8rem">{result['approval_probability']:.2%}</div><div class="label">approval probability</div>
            <div class="card-note" style="margin-top:1rem">Rejection probability: {result['rejection_probability']:.2%} · educational prototype, not a lending decision.</div></div>''',
            unsafe_allow_html=True,
        )


def render_batch() -> None:
    st.markdown('<div class="kicker">06 / BATCH LAB</div><div class="deck-title">Score a <span>dataset.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">Upload multiple applicants using the project schema, validate it, run the same inference path, and download scored results.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Applicant CSV", type=["csv"])
    if uploaded is None:
        st.markdown('<div class="callout">Required columns: ' + esc(", ".join(MODEL_INPUT_COLUMNS)) + '</div>', unsafe_allow_html=True)
        return
    data = pd.read_csv(uploaded)
    st.markdown(card("Rows loaded", f"{len(data):,}", "from uploaded CSV"), unsafe_allow_html=True)
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    st.dataframe(data.head(15), use_container_width=True, hide_index=True)
    if st.button("RUN BATCH PREDICTION", type="primary", use_container_width=True):
        with st.spinner("Running reusable inference pipeline…"):
            result = predict_batch(bundle, data)
        st.success(f"Completed {len(result):,} predictions.")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("DOWNLOAD CSV", result.to_csv(index=False).encode("utf-8"), "loan_predictions.csv", "text/csv", use_container_width=True)


def render_insights() -> None:
    st.markdown('<div class="kicker">07 / MODEL INSIGHTS</div><div class="deck-title">Open the <span>black box.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">Feature importance shows which encoded features the fitted XGBoost model relied on most in this experiment. It is not causal importance.</div>', unsafe_allow_html=True)
    section("Top model features")
    st.markdown(feature_bars(), unsafe_allow_html=True)
    section("Overfitting check")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(card("Decision Tree", "100% train", "89.28% test · clear generalization gap"), unsafe_allow_html=True)
    with c2:
        st.markdown(card("XGBoost", "97.68% train", "92.87% test · smaller gap, still needs validation"), unsafe_allow_html=True)
    st.markdown('<div class="callout warn" style="margin-top:1rem">Feature importance can describe model reliance, but it does not prove that a feature causes approval, is fair, or belongs in a production underwriting policy.</div>', unsafe_allow_html=True)


def render_architecture() -> None:
    st.markdown('<div class="kicker">08 / ARCHITECTURE</div><div class="deck-title">Notebook → <span>product.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">The repository separates the reusable ML core from the presentation layer so the same model logic can power the UI and batch workflow.</div>', unsafe_allow_html=True)
    section("Runtime architecture")
    st.markdown('''<div class="flow"><div class="flow-item"><strong>USER</strong><small>browser</small></div><div class="flow-item"><strong>STREAMLIT</strong><small>UI + routing</small></div><div class="flow-item"><strong>MODEL CORE</strong><small>src/model.py</small></div><div class="flow-item"><strong>PREPROCESS</strong><small>encode + scale</small></div><div class="flow-item"><strong>XGBOOST</strong><small>inference</small></div><div class="flow-item"><strong>RESULT</strong><small>decision + probability</small></div><div class="flow-item"><strong>CSV</strong><small>batch output</small></div></div>''', unsafe_allow_html=True)
    section("Repository responsibilities")
    cols = st.columns(4)
    items = [
        ("loan.ipynb", "Research", "EDA, experiments, benchmark evidence."),
        ("src/model.py", "ML core", "Training, preprocessing, prediction contract."),
        ("app.py", "Product UI", "Interactive website and live demo."),
        ("tests / CI", "Quality", "Regression checks and automated validation."),
    ]
    for col, (file, title, body) in zip(cols, items):
        with col:
            st.markdown(f'<div class="card"><div class="card-k">{esc(file)}</div><div style="font-weight:800;margin-top:.35rem">{esc(title)}</div><div class="card-note">{esc(body)}</div></div>', unsafe_allow_html=True)


def render_responsible() -> None:
    st.markdown('<div class="kicker">09 / RESPONSIBLE AI</div><div class="deck-title">A loan model affects <span>people.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="deck-sub">The responsible way to present this project is to make its boundaries visible, not hide them behind a score.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, title, body in [
        (cols[0], "Fairness", "No subgroup fairness audit has been completed."),
        (cols[1], "Calibration", "Probabilities have not been calibrated."),
        (cols[2], "Privacy", "Real production data governance is out of scope."),
        (cols[3], "Human oversight", "This prototype must not automate real lending decisions."),
    ]:
        with col:
            st.markdown(f'<div class="card"><div style="font-size:1rem;font-weight:800">{esc(title)}</div><div class="card-note">{esc(body)}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="callout bad" style="margin-top:1rem"><b>Boundary:</b> Loan Intelligence is an educational / portfolio ML prototype. Dataset representativeness, fairness, calibration, causal validity, drift, and production underwriting controls require separate validation.</div>', unsafe_allow_html=True)


def render_presentation() -> None:
    slides = [
        ("01 / OPENING", "Loan <span>Intelligence.</span>", "The project, as a product.", "A complete ML workflow that can be explored, challenged, and demonstrated live.", "92.87%", "held-out test accuracy"),
        ("02 / PROBLEM", "What are we actually <span>predicting?</span>", "Binary classification: loan status.", "Given applicant and loan characteristics, predict the target class. The objective is a traceable supervised-learning workflow, not a claim of perfect underwriting.", None, None),
        ("03 / DATA", "Meet the <span>data.</span>", "45K+ source rows · 13 raw model inputs.", f"The repository dataset contains {len(df):,} source rows and the documented applicant / loan features used by the model.", f"{len(df):,}", "source rows"),
        ("04 / PREPROCESSING", "Raw rows → <span>model-ready features.</span>", "Cleaning, encoding, scaling.", "loan_id is removed, selected numeric variables receive IQR clipping, sanity bounds are applied, categories are encoded, and RobustScaler is fitted on the training split.", "{0}".format(len(bundle.feature_columns)), "encoded features"),
        ("05 / IMBALANCE", "The target was <span>skewed.</span>", "SMOTETomek balances the training signal.", "SMOTE creates minority examples and Tomek links clean overlapping examples. The combined sampler is applied to training data only; the test set stays untouched.", "1 : 1", "post-resampling class balance"),
        ("06 / MODEL ARENA", "Six models. <span>One benchmark.</span>", "Comparable held-out evaluation.", "Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost were compared using accuracy plus class-1 precision, recall, and F1.", "6", "models tested"),
        ("07 / WINNER", "XGBoost reaches <span>92.87%.</span>", "Best overall documented benchmark result.", "XGBoost leads test accuracy, class-1 precision, and F1. Logistic Regression and SVM have higher class-1 recall, so the selection is a trade-off.", "0.83", "class-1 F1"),
        ("08 / ERRORS", "Where did the model <span>miss?</span>", "Confusion matrix over held-out data.", "True negatives, false positives, false negatives, and true positives tell a richer story than accuracy alone.", None, None),
        ("09 / GENERALIZATION", "Training score is not the story. <span>Test score is.</span>", "Overfitting is visible.", "Decision Tree reaches 100% train accuracy vs 89.28% test. XGBoost reaches 97.68% train vs 92.87% test. The remaining gap still needs validation.", "+", "generalization matters"),
        ("10 / LIVE LAB", "Stop the deck. <span>Run the model.</span>", "Interactive real inference.", "Change an applicant profile in Prediction Studio and execute the actual reusable XGBoost inference path.", "LIVE", "model connected"),
        ("11 / PRODUCT", "From notebook → <span>product.</span>", "Research, reusable ML core, UI, tests, CI.", "The notebook documents the experiment; src/model.py owns reusable ML logic; app.py turns it into an interactive product experience.", "03", "core workspaces"),
        ("12 / RESPONSIBLE AI", "A probability is not a <span>verdict.</span>", "Interpret output with context.", "Calibration, fairness, subgroup performance, drift, and production policy controls have not been established. This remains an educational prototype.", "NO", "production claim"),
        ("13 / PRODUCTION", "What would make this <span>production-grade?</span>", "The engineering roadmap continues.", "Version artifacts, add FastAPI, SHAP, calibration, subgroup evaluation, Docker, MLflow, registry, drift monitoring, observability, and CI/CD.", "NEXT", "engineering roadmap"),
        ("14 / CLOSE", "Research → Engineering → <span>Product.</span>", "The deliverable is bigger than one number.", "A reproducible pipeline, honest benchmark, reusable inference, batch workflow, documentation, and an interactive demo make Loan Intelligence defensible.", "LIVE", "demo ready"),
    ]
    if "slide" not in st.session_state:
        st.session_state.slide = 0
    index = st.session_state.slide
    meta, title, sub, body, stat, stat_label = slides[index]

    st.markdown(f'<div class="kicker">{esc(meta)} · INTERACTIVE DECK</div>', unsafe_allow_html=True)
    controls = st.columns([1, 1, 1, 1, 1, 1.5])
    with controls[0]:
        if st.button("← PREV", use_container_width=True):
            st.session_state.slide = (index - 1) % len(slides)
            st.rerun()
    with controls[1]:
        if st.button("NEXT →", use_container_width=True):
            st.session_state.slide = (index + 1) % len(slides)
            st.rerun()
    with controls[2]:
        if st.button("HOME", use_container_width=True):
            st.session_state.slide = 0
            st.rerun()
    with controls[3]:
        if st.button("MODEL", use_container_width=True):
            st.session_state.page = "Prediction Studio"
            st.rerun()
    with controls[4]:
        if st.button("INSIGHTS", use_container_width=True):
            st.session_state.page = "Model Insights"
            st.rerun()
    with controls[5]:
        st.markdown(f'<div style="text-align:right;padding:.7rem 0;color:#71849a;font:500 .57rem DM Mono">SLIDE {index+1:02d} / {len(slides):02d}</div>', unsafe_allow_html=True)

    st.markdown(
        f'''
        <div class="deck">
          <div class="deck-grid"></div><div class="deck-content">
            <div class="kicker">{esc(meta)}</div>
            <div class="deck-title">{title}</div>
            <div class="deck-sub">{esc(sub)}</div>
            <div class="deck-body">{esc(body)}</div>
            {f'<div class="deck-stat">{esc(stat)}</div><div class="label">{esc(stat_label)}</div>' if stat else ''}
            {confusion_matrix_html() if index == 7 else ''}
            {benchmark_bars() if index == 6 else ''}
            {chips(["INTERACTIVE", "REAL DATA", "XGBOOST", "RESPONSIBLE AI"]) if index in [0, 9, 11] else ''}
          </div>
          <div class="deck-footer"><div class="deck-progress"><div style="width:{(index+1)/len(slides)*100:.2f}%"></div></div><div class="deck-counter">{index+1}/{len(slides)}</div></div>
        </div>''',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
PAGES = [
    "Overview",
    "Data & Features",
    "Preprocessing",
    "Evaluation",
    "Prediction Studio",
    "Batch Lab",
    "Model Insights",
    "Architecture",
    "Responsible AI",
    "Presentation",
]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown('<div class="brand"><div class="kicker">◈ NTI / ML TRACK</div><div class="brand-title">LOAN <span>INTELLIGENCE</span></div><div class="brand-sub">AI-powered loan-status classification</div></div>', unsafe_allow_html=True)
    selected = st.radio("NAVIGATION", PAGES, index=PAGES.index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = selected
    st.markdown(
        '<div class="side-status"><span class="live-dot"></span><b>MODEL ONLINE</b><br>XGBoost · cached training<br>Test accuracy: <b>92.87%</b></div>'
        '<div class="side-links"><b>Keyboard-style deck:</b><br>PREV · NEXT · HOME · MODEL · INSIGHTS</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.caption("Educational / portfolio prototype — not real underwriting.")

if st.session_state.page == "Overview":
    render_overview()
elif st.session_state.page == "Data & Features":
    render_data()
elif st.session_state.page == "Preprocessing":
    render_preprocessing()
elif st.session_state.page == "Evaluation":
    render_evaluation()
elif st.session_state.page == "Prediction Studio":
    render_prediction()
elif st.session_state.page == "Batch Lab":
    render_batch()
elif st.session_state.page == "Model Insights":
    render_insights()
elif st.session_state.page == "Architecture":
    render_architecture()
elif st.session_state.page == "Responsible AI":
    render_responsible()
elif st.session_state.page == "Presentation":
    render_presentation()

st.markdown('<div class="footer-note">LOAN INTELLIGENCE · NTI MACHINE LEARNING · BETTER DATA → SMARTER MODELS → REAL IMPACT</div>', unsafe_allow_html=True)
