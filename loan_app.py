from pathlib import Path
import html
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

st.set_page_config(page_title="Loan Intelligence | NTI ML", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#03060b;--panel:#080e17;--line:#1a2b3f;--text:#f7fafc;--muted:#8796a9;--cyan:#22d3ee;--lime:#a3e635;--purple:#a78bfa;--red:#fb7185;--amber:#fbbf24}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(3,6,11,.94)}[data-testid="stSidebar"]{background:#050a11;border-right:1px solid var(--line)}
.block-container{max-width:1540px;padding:1rem 2rem 4rem}#MainMenu,footer{visibility:hidden}
.mono{font-family:'DM Mono',monospace}.eyebrow{color:var(--cyan);font-size:.62rem;font-weight:800;letter-spacing:.2em;text-transform:uppercase}.muted{color:var(--muted)}
.brand{padding:.1rem 0 1rem}.brand-row{display:flex;align-items:center;gap:.7rem}.brand-logo{width:42px;height:42px;border-radius:12px;background:white;padding:5px;object-fit:contain}.brand-mark{color:var(--cyan);font-weight:800;font-size:.72rem}.brand-title{font-size:1rem;font-weight:800}.brand-sub{font-size:.58rem;color:var(--muted);margin-top:.12rem}.nav-label{font-size:.58rem;color:#64748b;text-transform:uppercase;letter-spacing:.18em;font-weight:800;margin:1rem 0 .45rem}.team-mini{font-size:.59rem;color:#a3afbd;line-height:1.75;border-top:1px solid var(--line);padding-top:.75rem}.team-mini strong{color:#e5edf5}
.hero{position:relative;overflow:hidden;padding:2.8rem;border:1px solid var(--line);border-radius:30px;background:radial-gradient(circle at 91% 5%,rgba(34,211,238,.2),transparent 25%),radial-gradient(circle at 75% 115%,rgba(167,139,250,.15),transparent 32%),linear-gradient(135deg,#07111c,#06101a 56%,#0c1020);margin-bottom:1.25rem;box-shadow:0 25px 80px rgba(0,0,0,.25)}
.hero:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(to right,transparent 25%,black 70%);pointer-events:none}.hero:after{content:'';position:absolute;right:-100px;top:-160px;width:420px;height:420px;border:1px solid rgba(34,211,238,.14);border-radius:50%;box-shadow:0 0 0 35px rgba(34,211,238,.025),0 0 0 75px rgba(34,211,238,.012)}
.hero h1{position:relative;margin:.4rem 0 0;font-size:clamp(3rem,6vw,6rem);line-height:.86;letter-spacing:-.08em;font-weight:800}.hero h1 span,.page-title span{color:var(--cyan)}.hero p{position:relative;color:#aab8c8;max-width:900px;font-size:.9rem;line-height:1.8;margin:1rem 0 0}.hero-meta{position:relative;display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.4rem}.pill{border:1px solid var(--line);background:rgba(255,255,255,.035);padding:.45rem .72rem;border-radius:999px;color:#abb8c7;font-size:.61rem}.pill strong{color:#fff}
.page-title{font-size:2.5rem;font-weight:800;letter-spacing:-.07em;margin:.05rem 0 .2rem}.page-sub{color:var(--muted);font-size:.78rem;margin-bottom:1.25rem}.section-title{margin:1rem 0 .6rem;font-size:.61rem;text-transform:uppercase;letter-spacing:.18em;color:#8fa0b5;font-weight:800}
.card{border:1px solid var(--line);background:linear-gradient(145deg,#09111b,#070c14);border-radius:18px;padding:1.1rem;height:100%;box-shadow:0 12px 35px rgba(0,0,0,.14)}.card-k{color:#718198;font-size:.56rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800}.card-v{font-family:'DM Mono',monospace;font-size:1.55rem;margin-top:.3rem}.card-note{color:var(--muted);font-size:.63rem;margin-top:.3rem;line-height:1.5}
label,.stNumberInput label,.stSelectbox label,.stFileUploader label{color:#b7c1ce!important;font-size:.67rem!important;font-weight:600!important}input,[data-baseweb="select"]>div{background:#070d16!important;border-color:#1a2a3d!important;color:#f5f7fb!important;border-radius:10px!important}.stButton>button{border:1px solid rgba(34,211,238,.38);background:linear-gradient(135deg,#0c202a,#09151e);color:#e6fbff;border-radius:11px;min-height:43px;font-weight:800;transition:.2s}.stButton>button:hover{border-color:var(--cyan);color:#fff;transform:translateY(-1px);box-shadow:0 8px 25px rgba(34,211,238,.12)}.stDownloadButton>button{border-radius:11px}
.range-note{font-size:.56rem;color:#66768b;margin-top:-.35rem;margin-bottom:.55rem}.decision{min-height:305px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border:1px solid var(--line);border-radius:22px;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.12),transparent 48%),var(--panel);padding:1.5rem;box-shadow:0 20px 50px rgba(0,0,0,.2)}.decision .status{font-size:.58rem;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:800}.decision .label{margin-top:.45rem;font-size:2.8rem;line-height:1;font-weight:800;letter-spacing:-.06em}.approved{color:var(--lime)}.rejected{color:var(--red)}.prob{font-family:'DM Mono',monospace;margin-top:.8rem;font-size:1.05rem}.confidence-bar{width:82%;height:7px;background:#172232;border-radius:99px;overflow:hidden;margin-top:1rem}.confidence-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));border-radius:99px}.decision-note{color:var(--muted);font-size:.61rem;margin-top:.7rem}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:.7rem}.metric{border:1px solid var(--line);background:#090f19;border-radius:13px;padding:.8rem}.metric .k{color:var(--muted);font-size:.55rem;text-transform:uppercase;letter-spacing:.1em}.metric .v{font-family:'DM Mono',monospace;font-size:1rem;margin-top:.2rem}.risk{padding:.65rem .75rem;border-left:2px solid var(--lime);background:rgba(163,230,53,.035);margin:.35rem 0;border-radius:0 9px 9px 0;color:#aeb9c8;font-size:.65rem;line-height:1.45}.feature-chip{display:inline-block;border:1px solid #1d3348;background:#09121c;border-radius:999px;padding:.35rem .55rem;margin:.18rem;color:#b7c5d5;font-family:'DM Mono',monospace;font-size:.56rem}.feature-chip b{color:var(--cyan)}
.insight{border:1px solid var(--line);border-radius:16px;padding:1rem;background:linear-gradient(145deg,#09121c,#070d15);margin-bottom:.7rem}.insight h4{margin:0;font-size:.75rem}.insight p{margin:.3rem 0 0;color:var(--muted);font-size:.65rem;line-height:1.6}.about-hero{border:1px solid var(--line);border-radius:24px;padding:2rem;background:radial-gradient(circle at 90% 10%,rgba(167,139,250,.13),transparent 30%),#080e17}.team-card{border:1px solid var(--line);border-radius:16px;background:#090f18;padding:1rem;height:100%}.team-number{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.62rem}.team-name{font-size:.79rem;font-weight:700;margin-top:.35rem}.team-role{color:var(--muted);font-size:.6rem;margin-top:.25rem}.footer-note{text-align:center;color:#536176;font-size:.56rem;margin-top:2rem;letter-spacing:.06em}
.deck{border:1px solid #1b3044;border-radius:28px;overflow:hidden;background:radial-gradient(circle at 88% 8%,rgba(34,211,238,.14),transparent 25%),radial-gradient(circle at 18% 95%,rgba(167,139,250,.09),transparent 30%),#03070d;padding:2.2rem 2.4rem;min-height:590px;position:relative}.deck-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.02) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.02) 1px,transparent 1px);background-size:38px 38px;mask-image:linear-gradient(to bottom,black,transparent);pointer-events:none}.deck-content{position:relative;z-index:1}.deck-kicker{font:800 .62rem 'DM Mono',monospace;letter-spacing:.18em;color:var(--cyan)}.deck h2{font-size:clamp(2.3rem,5vw,5rem);line-height:.92;letter-spacing:-.075em;margin:.9rem 0 1rem;max-width:1050px}.deck h2 span{color:var(--cyan)}.deck-sub{font-size:1rem;color:#d6e0e9;max-width:900px;line-height:1.55}.deck-body{color:#8fa1b4;max-width:900px;line-height:1.75;font-size:.78rem;margin-top:1rem}.deck-stat{font:500 clamp(3.5rem,7vw,7rem) 'DM Mono',monospace;color:var(--cyan);letter-spacing:-.08em;line-height:.9;margin:1rem 0 .35rem}.deck-label{font:800 .58rem 'DM Mono',monospace;color:#64778d;letter-spacing:.16em;text-transform:uppercase}.deck-chips{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:1.2rem}.deck-chip{border:1px solid #20364b;background:rgba(255,255,255,.035);border-radius:999px;padding:.42rem .62rem;color:#aebdcb;font:500 .56rem 'DM Mono',monospace}.quote{border-left:2px solid var(--cyan);padding-left:14px;color:#d4e0e9;max-width:820px;margin-top:1.2rem;font-size:.82rem;line-height:1.7}.deck-footer{position:absolute;bottom:1.4rem;left:2.4rem;right:2.4rem;display:flex;gap:.8rem;align-items:center;z-index:2}.deck-progress{height:3px;flex:1;background:#162536;border-radius:99px;overflow:hidden}.deck-progress>div{height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));transition:.35s}.bar-row{display:grid;grid-template-columns:180px 1fr 65px;gap:10px;align-items:center;margin:.55rem 0}.bar-name{font-size:.62rem;color:#b9c6d3}.bar-track{height:9px;background:#142132;border-radius:99px;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,#22d3ee,#a3e635);border-radius:99px}.bar-value{font:500 .62rem 'DM Mono',monospace;text-align:right;color:#e8f4f8}.cm-grid{display:grid;grid-template-columns:100px 1fr 1fr;gap:6px;max-width:620px;margin-top:1rem}.cm-grid>div{padding:18px 10px;text-align:center;border:1px solid #1b3044;border-radius:10px;background:#08111b;font-family:'DM Mono',monospace}.cm-grid .head{font-size:.55rem;color:#6f8194;background:transparent;border:0}.cm-grid .tn{color:#22d3ee;font-size:1.2rem}.cm-grid .tp{color:#a3e635;font-size:1.2rem}.cm-grid .fp{color:#fb7185;font-size:1.2rem}.cm-grid .fn{color:#fbbf24;font-size:1.2rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model():
    return train_model(DATA_PATH)

@st.cache_data(show_spinner=False)
def get_reference():
    return pd.read_csv(DATA_PATH)

if not DATA_PATH.exists():
    st.error("loan_data.csv was not found in the project root.")
    st.stop()

with st.spinner("Booting Loan Intelligence…"):
    bundle, metrics = get_model()
reference = get_reference()

BENCHMARK = pd.DataFrame([
    ["Logistic Regression",86.55,64,92,75],["KNN",86.29,64,88,74],
    ["Decision Tree",89.28,73,82,77],["Random Forest",89.48,71,88,79],
    ["SVM",88.02,67,92,77],["XGBoost",92.87,87,80,83]
], columns=["Model","Accuracy","Precision","Recall","F1"])
RANGES = {
    "income": (float(reference.person_income.min()), float(reference.person_income.max())),
    "loan_amount": (float(reference.loan_amnt.min()), float(reference.loan_amnt.max())),
    "interest": (float(reference.loan_int_rate.min()), float(reference.loan_int_rate.max())),
    "loan_percent_income": (float(reference.loan_percent_income.min()), min(1.0, float(reference.loan_percent_income.max()))),
    "credit_history": (float(reference.cb_person_cred_hist_length.min()), float(reference.cb_person_cred_hist_length.max())),
    "credit_score": (float(reference.credit_score.min()), float(reference.credit_score.max())),
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def card(k, v, note=""):
    return f'<div class="card"><div class="card-k">{html.escape(str(k))}</div><div class="card-v">{html.escape(str(v))}</div><div class="card-note">{html.escape(str(note))}</div></div>'

def num(label, lo, hi, default, step, key):
    lo, hi, default, step = float(lo), float(hi), float(default), float(step)
    default = clamp(default, lo, hi)
    return st.number_input(label, min_value=lo, max_value=hi, value=default, step=step, key=key)

def render_bars(df, value_col="Accuracy", highlight="XGBoost"):
    rows=[]
    for _, r in df.iterrows():
        cls="bar-fill" if r["Model"]==highlight else "bar-fill"
        rows.append(f'<div class="bar-row"><div class="bar-name">{html.escape(r["Model"])}</div><div class="bar-track"><div class="{cls}" style="width:{float(r[value_col]):.2f}%"></div></div><div class="bar-value">{float(r[value_col]):.2f}%</div></div>')
    return ''.join(rows)

def slide_data():
    return [
        ("OPENING","Loan <span>Intelligence.</span>","An end-to-end Machine Learning system for binary loan-status prediction.","NTI Machine Learning track final project connecting data preparation, class-imbalance handling, model benchmarking, XGBoost inference, batch scoring, and a deployable product experience.","92.87%","HELD-OUT TEST ACCURACY",["MACHINE LEARNING","BINARY CLASSIFICATION","XGBOOST","NTI"]),
        ("PROBLEM","Turn applicant signals into a <span>repeatable decision.</span>","Loan status is framed as a binary classification problem.","The system receives applicant and loan attributes, applies the project's preprocessing path, and returns a predicted class plus probability estimates. The goal is a traceable workflow rather than an unexplained score.",None,None,["INPUT → TRANSFORM → PREDICT","REPEATABLE PIPELINE"]),
        ("DATASET",f"<span>{len(reference):,}</span> rows. 13 raw inputs.","Applicant, employment, loan, credit, and target information in one supervised-learning table.","The raw table includes demographic attributes, income, employment experience, home ownership, loan intent, loan amount, interest rate, loan-to-income ratio, credit history, credit score, previous default history, and loan status.",f"{len(reference):,}","SOURCE ROWS",["NUMERIC + CATEGORICAL","13 RAW INPUTS","LOAN_STATUS TARGET"]),
        ("DATA QUALITY","Clean the signal. <span>Expose the assumptions.</span>","EDA is part of the model story, not decoration.","The workflow removes loan_id from modeling, applies IQR-based clipping to selected numeric variables, and enforces the project constraints of age ≤ 80 and employment experience ≤ 60 before encoding and scaling.",None,None,["EDA","IQR CLIPPING","BOUNDS","QUALITY CHECKS"]),
        ("PREPROCESSING","From raw records to <span>model-ready features.</span>","Categorical information becomes numerical representation.","Gender and previous-default history are binary mapped. Education is ordinally encoded. Home ownership and loan intent are one-hot encoded with drop_first=True. RobustScaler transforms the feature space before model fitting.",None,None,["BINARY MAP","ORDINAL MAP","ONE-HOT","ROBUSTSCALER"]),
        ("IMBALANCE","The training target was <span>imbalanced.</span>","The minority class needed explicit treatment during training.","SMOTETomek is applied only to the training data after scaling. SMOTE synthesizes minority examples while Tomek links clean borderline pairs. The held-out test set remains untouched for evaluation.","27,887 × 2","BALANCED TRAINING ROWS",["SMOTE","TOMEK LINKS","TRAINING ONLY"]),
        ("MODEL ARENA","Six classifiers entered the <span>benchmark.</span>","Model selection is evidence-driven.","Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost were compared on the same held-out test split. Accuracy is shown alongside class-1 precision, recall, and F1.",None,None,["LOGISTIC","KNN","TREE","RANDOM FOREST","SVM","XGBOOST"]),
        ("WINNER","XGBoost led the benchmark at <span>92.87%.</span>","Strongest overall held-out accuracy, class-1 precision, and class-1 F1 in this benchmark.","This does not mean XGBoost is universally best. It means it was the strongest option under this experiment, dataset, split, preprocessing path, and metric comparison.","92.87%","XGBOOST TEST ACCURACY",["87% CLASS-1 PRECISION","80% CLASS-1 RECALL","83% CLASS-1 F1"]),
        ("GENERALIZATION","A good score still needs <span>skepticism.</span>","The benchmark contains a visible train/test gap.","The Decision Tree reaches 100% training accuracy with a lower test result, indicating strong overfitting. XGBoost also has a train/test gap, so cross-validation, tuning, calibration, and threshold analysis are natural next steps.",None,None,["OVERFITTING AWARENESS","HELD-OUT TEST","NEXT: CROSS-VALIDATION"]),
        ("ARCHITECTURE","One flow from <span>CSV to decision.</span>","Data → cleaning → encoding → scaling → SMOTETomek → XGBoost → inference.","The same reusable model bundle powers single-applicant inference and batch CSV scoring. The Streamlit layer is presentation and interaction; src/model.py owns the ML workflow.",None,None,["DATA","FEATURE ENGINEERING","MODEL","UI","BATCH INFERENCE"]),
        ("PRODUCT","Prediction Studio is the <span>live front door.</span>","All 13 raw model inputs are exposed with dataset-derived numeric guardrails.","Users can enter an applicant, run the real trained XGBoost bundle, and inspect predicted status, approval probability, rejection probability, and benchmark context. Numeric arguments are normalized to consistent float types to avoid Streamlit mixed-type failures.",None,None,["13 INPUTS","RANGE-GUARDED","LIVE INFERENCE"]),
        ("BATCH LAB","One applicant or <span>an entire CSV.</span>","The demo becomes a reusable inference utility.","Download the schema template, upload raw applicant rows, validate required columns, score them with the same model bundle, inspect approved/rejected counts, and download the scored CSV. loan_status can be supplied for row-level accuracy evaluation.",None,None,["UPLOAD","VALIDATE","SCORE","DOWNLOAD"]),
        ("MODEL INSIGHTS","Don't just show the answer. <span>Show the evidence.</span>","Diagnostics make the model choice defendable.","The product exposes the benchmark, feature importance, confusion matrix, and precision/recall trade-off. Feature importance is model-specific evidence, not causal explanation.",None,None,["BENCHMARK","FEATURE IMPORTANCE","CONFUSION MATRIX"]),
        ("LIMITATIONS","Good ML work includes <span>what could be better.</span>","This is an educational prototype, not a production credit-decision engine.","The results are dataset-specific. IQR bounds are computed before the split in the current workflow, education uses an ordinal assumption, and probability outputs are not calibrated financial risk. Production use would require fairness, governance, validation, and monitoring.",None,None,["DATASET-SPECIFIC","CALIBRATION","FAIRNESS REVIEW","LEAKAGE HARDENING"]),
        ("ROADMAP","From student project to <span>ML service.</span>","The next gains come from stronger validation and engineering.","Next: fit preprocessing statistics inside training folds, tune XGBoost with validation, calibrate probabilities, add cross-validation and threshold analysis, version data and models, monitor drift, and expose controlled inference APIs with audit logging.",None,None,["CV","CALIBRATION","MLOPS","MONITORING","API"]),
        ("CLOSING","Four people. <span>One ML system.</span>","NTI Machine Learning Track — Loan Status Prediction.","The final experience combines analysis, preprocessing, benchmarking, inference, batch scoring, diagnostics, documentation, and presentation into one coherent system.","4","TEAM MEMBERS",TEAM),
    ]

def render_presentation():
    slides=slide_data()
    if "slide" not in st.session_state: st.session_state.slide=0
    n=len(slides)
    a,b,c,d=st.columns([.7,.7,4,1.2])
    with a:
        if st.button("← PREV", use_container_width=True): st.session_state.slide=max(0,st.session_state.slide-1)
    with b:
        if st.button("NEXT →", use_container_width=True): st.session_state.slide=min(n-1,st.session_state.slide+1)
    with c:
        st.progress((st.session_state.slide+1)/n)
    with d:
        st.markdown(f'<div class="mono" style="text-align:right;color:#7e91a5;font-size:.65rem;padding-top:.65rem">{st.session_state.slide+1:02d} / {n:02d}</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Presentation mode · use the buttons or jump directly</div>',unsafe_allow_html=True)
    jump=st.selectbox("Jump to slide",[f"{i+1:02d} · {s[0]}" for i,s in enumerate(slides)],index=st.session_state.slide,label_visibility="collapsed")
    chosen=int(jump.split(" ")[0])-1
    if chosen!=st.session_state.slide: st.session_state.slide=chosen; st.rerun()
    k,title,sub,body,stat,label,chips=slides[st.session_state.slide]
    chips_html=''.join(f'<span class="deck-chip">{html.escape(str(x))}</span>' for x in chips)
    stat_html=f'<div class="deck-stat">{html.escape(stat)}</div><div class="deck-label">{html.escape(label)}</div>' if stat else ''
    quote_html=''
    if st.session_state.slide==1: quote_html='<div class="quote">A strong demo is not only a prediction. It is a traceable ML workflow that can be inspected, tested, and defended.</div>'
    st.markdown(f'<div class="deck"><div class="deck-grid"></div><div class="deck-content"><div class="deck-kicker">{k} · NTI MACHINE LEARNING</div><h2>{title}</h2><div class="deck-sub">{sub}</div>{stat_html}<div class="deck-body">{body}</div>{quote_html}<div class="deck-chips">{chips_html}</div></div><div class="deck-footer"><div class="deck-progress"><div style="width:{(st.session_state.slide+1)/n*100:.1f}%"></div></div><div class="mono" style="font-size:.56rem;color:#64778d">LOAN INTELLIGENCE</div></div></div>',unsafe_allow_html=True)

    s=st.session_state.slide
    if s==6:
        st.markdown('<div class="section-title">Interactive benchmark visualization</div>',unsafe_allow_html=True)
        st.markdown(render_bars(BENCHMARK),unsafe_allow_html=True)
    elif s==7:
        st.markdown('<div class="section-title">Why the winner is defendable</div>',unsafe_allow_html=True)
        x,y,z=st.columns(3)
        x.markdown(card('Accuracy','92.87%','Best in this benchmark'),unsafe_allow_html=True)
        y.markdown(card('Precision','87%','Class 1 / XGBoost'),unsafe_allow_html=True)
        z.markdown(card('F1','83%','Class 1 / XGBoost'),unsafe_allow_html=True)
    elif s==9:
        st.markdown('<div class="section-title">Architecture map</div>',unsafe_allow_html=True)
        st.markdown('<div class="card" style="font-family:DM Mono,monospace;text-align:center;padding:1.5rem">CSV / APPLICANT → CLEAN + ENCODE → ROBUSTSCALER → SMOTETOMEK (TRAIN) → XGBOOST → PREDICTION → UI / BATCH CSV</div>',unsafe_allow_html=True)
    elif s==10:
        st.markdown('<div class="section-title">Live model lab — real inference inside the presentation</div>',unsafe_allow_html=True)
        q1,q2=st.columns(2)
        with q1:
            age=st.number_input('Age',18,80,30,1,key='pres_age')
            income=num('Annual income',*RANGES['income'],60000,1000,'pres_income')
            emp=st.number_input('Employment experience (years)',0,60,5,1,key='pres_emp')
            edu=st.selectbox('Education',['High School','Associate','Bachelor','Master','Doctorate'],index=2,key='pres_edu')
            gender=st.selectbox('Gender',['male','female'],key='pres_gender')
            home=st.selectbox('Home ownership',['RENT','OWN','MORTGAGE','OTHER'],key='pres_home')
        with q2:
            intent=st.selectbox('Loan intent',['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT'],key='pres_intent')
            amount=num('Loan amount',*RANGES['loan_amount'],10000,500,'pres_amount')
            rate=num('Interest rate (%)',*RANGES['interest'],10,.1,'pres_rate')
            ratio=num('Loan / income ratio',*RANGES['loan_percent_income'],.2,.01,'pres_ratio')
            history=num('Credit history length (years)',*RANGES['credit_history'],4,.5,'pres_history')
            score=num('Credit score',*RANGES['credit_score'],680,1,'pres_score')
            default=st.selectbox('Previous loan default',['No','Yes'],key='pres_default')
        if st.button('RUN LIVE XGBOOST →',use_container_width=True,type='primary'):
            applicant={'person_age':age,'person_income':income,'person_home_ownership':home,'person_emp_exp':emp,'loan_intent':intent,'loan_amnt':amount,'loan_int_rate':rate,'loan_percent_income':ratio,'cb_person_cred_hist_length':history,'credit_score':score,'previous_loan_defaults_on_file':default,'person_gender':gender,'person_education':edu}
            st.session_state.pres_result=predict_one(bundle,applicant)
        if st.session_state.get('pres_result'):
            r=st.session_state.pres_result; cls='approved' if r['prediction']==1 else 'rejected'
            st.markdown(f'<div class="decision"><div class="status">LIVE MODEL OUTPUT</div><div class="label {cls}">{r["label"].upper()}</div><div class="prob">Approval probability · {r["approval_probability"]:.1%}</div><div class="confidence-bar"><div class="confidence-fill" style="width:{r["approval_probability"]*100:.1f}%"></div></div></div>',unsafe_allow_html=True)
    elif s==11:
        st.markdown('<div class="section-title">Batch scoring is also available without leaving the deck</div>',unsafe_allow_html=True)
        st.info('Use the Batch Lab page for the full upload workflow: template download → schema validation → XGBoost scoring → scored CSV download.')
    elif s==12:
        st.markdown('<div class="section-title">XGBoost feature importance — top 10</div>',unsafe_allow_html=True)
        imp=pd.Series(bundle.model.feature_importances_,index=bundle.feature_columns).sort_values(ascending=False).head(10).sort_values()
        max_imp=float(imp.max()) if len(imp) else 1.0
        for name,val in imp.items():
            width=float(val)/max_imp*100
            st.markdown(f'<div class="bar-row"><div class="bar-name">{html.escape(name)}</div><div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div><div class="bar-value">{float(val):.3f}</div></div>',unsafe_allow_html=True)
        st.caption('Model-specific feature importance is an internal ranking signal; it is not a causal explanation.')
    elif s==13:
        cm=metrics['confusion_matrix']
        st.markdown('<div class="section-title">Held-out confusion matrix</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="cm-grid"><div class="head"></div><div class="head">PRED 0</div><div class="head">PRED 1</div><div class="head">ACTUAL 0</div><div class="tn">{cm[0][0]:,}</div><div class="fp">{cm[0][1]:,}</div><div class="head">ACTUAL 1</div><div class="fn">{cm[1][0]:,}</div><div class="tp">{cm[1][1]:,}</div></div>',unsafe_allow_html=True)
    elif s==15:
        cols=st.columns(4)
        for i,(col,name) in enumerate(zip(cols,TEAM)): col.markdown(f'<div class="team-card"><div class="team-number">0{i+1}</div><div class="team-name">{html.escape(name)}</div><div class="team-role">NTI Machine Learning Track</div></div>',unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown(f'<div class="brand"><div class="brand-row"><img class="brand-logo" src="{NTI_LOGO}"><div><div class="brand-mark">NTI / ML</div><div class="brand-title">Loan Intelligence</div><div class="brand-sub">Interactive classification system</div></div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Workspace</div>',unsafe_allow_html=True)
    page=st.radio('Navigate',['Prediction Studio','Batch Lab','Project Presentation','Model Insights','Team & About'],label_visibility='collapsed')
    st.markdown('<div class="nav-label">Live system</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="risk"><strong>XGBoost</strong><br>{metrics["accuracy"]:.2%} held-out test accuracy</div><div class="risk"><strong>{len(reference):,} source rows</strong><br>{len(bundle.feature_columns)} encoded features</div><div class="risk"><strong>SMOTETomek</strong><br>{metrics["train_rows_after_smotetomek"]:,} balanced training rows</div>',unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Team</div>',unsafe_allow_html=True)
    st.markdown('<div class="team-mini">'+'<br>'.join(f'<strong>{i+1}.</strong> {html.escape(n)}' for i,n in enumerate(TEAM))+'</div>',unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:.8rem;font-size:.58rem"><a href="{NTI_SITE}" target="_blank" style="color:#22d3ee;text-decoration:none">Official NTI website ↗</a></div>',unsafe_allow_html=True)

if page=='Prediction Studio':
    st.markdown(f'<div class="hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK / FINAL PROJECT</div><h1>Loan <span>Intelligence.</span></h1><p>Interactive binary loan-status prediction using the reusable XGBoost workflow. Numeric controls are bounded by the reference dataset and all numeric arguments use consistent float types for Streamlit compatibility.</p><div class="hero-meta"><div class="pill"><strong>{metrics["accuracy"]:.2%}</strong> test accuracy</div><div class="pill"><strong>{len(reference):,}</strong> source rows</div><div class="pill"><strong>{len(bundle.feature_columns)}</strong> encoded features</div><div class="pill"><strong>6</strong> classifiers benchmarked</div></div></div>',unsafe_allow_html=True)
    left,right=st.columns([1.18,.82],gap='large')
    with left:
        st.markdown('<div class="section-title">01 / Applicant profile — all 13 model inputs</div>',unsafe_allow_html=True)
        with st.container(border=True):
            a,b=st.columns(2)
            with a:
                age=st.number_input('Age',18,80,30,1,key='main_age')
                income=num('Annual income',*RANGES['income'],60000,1000,'main_income')
                emp=st.number_input('Employment experience (years)',0,60,5,1,key='main_emp')
                edu=st.selectbox('Education',['High School','Associate','Bachelor','Master','Doctorate'],index=2,key='main_edu')
                gender=st.selectbox('Gender',['male','female'],key='main_gender')
                home=st.selectbox('Home ownership',['RENT','OWN','MORTGAGE','OTHER'],key='main_home')
                default=st.selectbox('Previous loan default',['No','Yes'],key='main_default')
            with b:
                intent=st.selectbox('Loan intent',['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT'],key='main_intent')
                amount=num('Loan amount',*RANGES['loan_amount'],10000,500,'main_amount')
                rate=num('Interest rate (%)',*RANGES['interest'],10,.1,'main_rate')
                ratio=num('Loan / income ratio',*RANGES['loan_percent_income'],.2,.01,'main_ratio')
                history=num('Credit history length (years)',*RANGES['credit_history'],4,.5,'main_history')
                score=num('Credit score',*RANGES['credit_score'],680,1,'main_score')
            run=st.button('RUN LOAN INTELLIGENCE →',use_container_width=True,type='primary')
    with right:
        st.markdown('<div class="section-title">02 / Decision output</div>',unsafe_allow_html=True)
        if run:
            applicant={'person_age':age,'person_income':income,'person_home_ownership':home,'person_emp_exp':emp,'loan_intent':intent,'loan_amnt':amount,'loan_int_rate':rate,'loan_percent_income':ratio,'cb_person_cred_hist_length':history,'credit_score':score,'previous_loan_defaults_on_file':default,'person_gender':gender,'person_education':edu}
            st.session_state.last_prediction=predict_one(bundle,applicant)
        result=st.session_state.get('last_prediction')
        if result:
            cls='approved' if result['prediction']==1 else 'rejected';p=result['approval_probability']
            st.markdown(f'<div class="decision"><div class="status">MODEL DECISION</div><div class="label {cls}">{result["label"].upper()}</div><div class="prob">Approval probability · {p:.1%}</div><div class="confidence-bar"><div class="confidence-fill" style="width:{p*100:.1f}%"></div></div><div class="decision-note">XGBoost probability output — not a calibrated financial-risk score.</div></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="metric-row"><div class="metric"><div class="k">Approved class</div><div class="v">{p:.1%}</div></div><div class="metric"><div class="k">Rejected class</div><div class="v">{result["rejection_probability"]:.1%}</div></div><div class="metric"><div class="k">Test accuracy</div><div class="v">{metrics["accuracy"]:.1%}</div></div></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="decision"><div class="status">READY FOR INFERENCE</div><div class="label" style="color:#c7d2df">AWAITING INPUT</div><div class="prob">Fill the profile and run the model.</div><div class="confidence-bar"><div class="confidence-fill" style="width:0%"></div></div><div class="decision-note">The exact trained feature layout will be used.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">03 / System pulse</div>',unsafe_allow_html=True)
    x,y,z=st.columns(3)
    x.markdown(card('Best benchmark','XGBoost','Highest test accuracy in the six-model comparison'),unsafe_allow_html=True)
    y.markdown(card('Class-1 F1','0.83','XGBoost on held-out test data'),unsafe_allow_html=True)
    z.markdown(card('Training balance','SMOTETomek','Applied only to training data'),unsafe_allow_html=True)

elif page=='Batch Lab':
    st.markdown('<div class="page-title">Batch <span>Lab.</span></div><div class="page-sub">Upload raw applicant records, validate the schema, score them, and export the results.</div>',unsafe_allow_html=True)
    template_cols=MODEL_INPUT_COLUMNS+['loan_id','loan_status']
    template=pd.DataFrame([{c:'' for c in template_cols}])
    defaults={'person_age':30,'person_income':60000,'person_home_ownership':'RENT','person_emp_exp':5,'loan_intent':'EDUCATION','loan_amnt':10000,'loan_int_rate':10.0,'loan_percent_income':.2,'cb_person_cred_hist_length':4.0,'credit_score':680,'previous_loan_defaults_on_file':'No','person_gender':'male','person_education':'Bachelor','loan_id':'example_001'}
    for k,v in defaults.items(): template.loc[0,k]=v
    l,r=st.columns([.75,1.25],gap='large')
    with l:
        st.markdown('<div class="section-title">01 / Schema</div>',unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-k">Required raw fields</div><div style="margin-top:.5rem">'+''.join(f'<span class="feature-chip"><b>•</b> {html.escape(c)}</span>' for c in MODEL_INPUT_COLUMNS)+'</div><div class="card-note">loan_id is optional. loan_status is optional and only used for evaluation when present.</div></div>',unsafe_allow_html=True)
        st.download_button('DOWNLOAD CSV TEMPLATE',template.to_csv(index=False).encode(),file_name='loan_prediction_template.csv',mime='text/csv',use_container_width=True)
        upload=st.file_uploader('Upload applicant CSV',type=['csv'])
    with r:
        if upload is None:
            st.markdown('<div class="about-hero"><div class="eyebrow mono">BATCH INFERENCE</div><h2 style="font-size:2rem;letter-spacing:-.05em">Score many applicants in one run.</h2><p class="muted">Upload the template or a CSV containing the 13 required raw model inputs.</p><div class="risk">The schema is validated before inference. Missing columns are rejected instead of silently producing partial predictions.</div></div>',unsafe_allow_html=True)
        else:
            try:
                df=pd.read_csv(upload); missing=[c for c in MODEL_INPUT_COLUMNS if c not in df.columns]
                if missing: st.error('Missing required columns: '+', '.join(missing))
                else:
                    st.success(f'Validated {len(df):,} rows.')
                    st.dataframe(df.head(8),use_container_width=True)
                    if st.button('RUN BATCH INFERENCE →',use_container_width=True,type='primary'):
                        try: st.session_state.batch_scored=predict_batch(bundle,df);st.success(f'Scored {len(st.session_state.batch_scored):,} rows.')
                        except Exception as e: st.error(f'Inference failed: {e}')
            except Exception as e: st.error(f'Could not read CSV: {e}')
    scored=st.session_state.get('batch_scored')
    if scored is not None:
        st.markdown('<div class="section-title">02 / Results</div>',unsafe_allow_html=True)
        a,b,c=st.columns(3);a.markdown(card('Approved',int((scored.prediction==1).sum()),'Predicted rows'),unsafe_allow_html=True);b.markdown(card('Rejected',int((scored.prediction==0).sum()),'Predicted rows'),unsafe_allow_html=True);c.markdown(card('Rows scored',len(scored),'Inference output'),unsafe_allow_html=True)
        if 'actual_status' in scored.columns: st.info(f'Batch accuracy against supplied loan_status: {float(scored.correct.mean()):.2%}')
        st.dataframe(scored,use_container_width=True)
        st.download_button('DOWNLOAD SCORED CSV',scored.to_csv(index=False).encode(),file_name='loan_predictions_scored.csv',mime='text/csv',use_container_width=True)

elif page=='Project Presentation':
    st.markdown('<div class="page-title">Project <span>Presentation.</span></div><div class="page-sub">A full technical deck inside the product — with live charts, model diagnostics, and real inference.</div>',unsafe_allow_html=True)
    render_presentation()

elif page=='Model Insights':
    st.markdown('<div class="page-title">Model <span>Insights.</span></div><div class="page-sub">The evidence behind the final model choice.</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.markdown(card('Accuracy','92.87%','XGBoost held-out test'),unsafe_allow_html=True);b.markdown(card('Class-1 precision','87%','XGBoost'),unsafe_allow_html=True);c.markdown(card('Class-1 recall','80%','XGBoost'),unsafe_allow_html=True);d.markdown(card('Class-1 F1','83%','XGBoost'),unsafe_allow_html=True)
    st.markdown('<div class="section-title">Six-model benchmark</div>',unsafe_allow_html=True)
    st.markdown(render_bars(BENCHMARK),unsafe_allow_html=True)
    st.dataframe(BENCHMARK.style.format({'Accuracy':'{:.2f}%','Precision':'{:.0f}%','Recall':'{:.0f}%','F1':'{:.0f}%'}),use_container_width=True,hide_index=True)
    l,r=st.columns(2,gap='large')
    with l:
        st.markdown('<div class="section-title">Top XGBoost feature importance</div>',unsafe_allow_html=True)
        imp=pd.Series(bundle.model.feature_importances_,index=bundle.feature_columns).sort_values(ascending=False).head(10).sort_values()
        max_imp=float(imp.max()) if len(imp) else 1.0
        for name,val in imp.items():
            st.markdown(f'<div class="bar-row"><div class="bar-name">{html.escape(name)}</div><div class="bar-track"><div class="bar-fill" style="width:{float(val)/max_imp*100:.1f}%"></div></div><div class="bar-value">{float(val):.3f}</div></div>',unsafe_allow_html=True)
    with r:
        cm=metrics['confusion_matrix']
        st.markdown('<div class="section-title">Held-out confusion matrix</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="cm-grid"><div class="head"></div><div class="head">PRED 0</div><div class="head">PRED 1</div><div class="head">ACTUAL 0</div><div class="tn">{cm[0][0]:,}</div><div class="fp">{cm[0][1]:,}</div><div class="head">ACTUAL 1</div><div class="fn">{cm[1][0]:,}</div><div class="tp">{cm[1][1]:,}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="insight"><h4>Precision / recall trade-off</h4><p>XGBoost has stronger class-1 precision and F1, while Logistic Regression and SVM have higher class-1 recall. The selection reflects the benchmark objective used here.</p></div>',unsafe_allow_html=True)
        st.markdown('<div class="insight"><h4>Generalization</h4><p>The Decision Tree reaches 100% training accuracy and a lower test result. XGBoost also has a train/test gap, so further validation remains important.</p></div>',unsafe_allow_html=True)

else:
    st.markdown(f'<div class="about-hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK</div><h1 style="font-size:3.3rem;letter-spacing:-.07em;margin:.5rem 0">Loan Intelligence <span style="color:#22d3ee">Project.</span></h1><p style="color:#aab8c8;max-width:850px;line-height:1.8">A complete educational ML product combining exploratory analysis, preprocessing, imbalance handling, model benchmarking, XGBoost inference, batch scoring, diagnostics, and an interactive presentation layer.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Team</div>',unsafe_allow_html=True)
    cols=st.columns(4); roles=['ML workflow & integration','Data analysis & preprocessing','Model evaluation & experimentation','Application & presentation']
    for i,(col,name,role) in enumerate(zip(cols,TEAM,roles)): col.markdown(f'<div class="team-card"><div class="team-number">0{i+1}</div><div class="team-name">{html.escape(name)}</div><div class="team-role">{role}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">System at a glance</div>',unsafe_allow_html=True)
    cols=st.columns(4)
    for col,k,v,note in zip(cols,['DATA','FEATURES','MODELS','WINNER'],[f'{len(reference):,}',str(len(bundle.feature_columns)),'6','92.87%'],['source rows','encoded features','benchmarked classifiers','XGBoost test accuracy']): col.markdown(card(k,v,note),unsafe_allow_html=True)
    st.markdown('<div class="section-title">Technical stack</div>',unsafe_allow_html=True)
    st.markdown(''.join(f'<span class="feature-chip">{x}</span>' for x in ['Python','Pandas','Scikit-learn','imbalanced-learn','XGBoost','Streamlit','GitHub Actions']),unsafe_allow_html=True)
    st.markdown(f'<div class="footer-note">Educational ML prototype · <a href="{NTI_SITE}" target="_blank" style="color:#22d3ee">Official NTI website</a></div>',unsafe_allow_html=True)
