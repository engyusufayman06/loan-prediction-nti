from pathlib import Path
import html
import json

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
[data-testid="stHeader"]{background:rgba(3,6,11,.9)}[data-testid="stSidebar"]{background:#050a11;border-right:1px solid var(--line)}
.block-container{max-width:1540px;padding:1rem 2rem 4rem}#MainMenu,footer{visibility:hidden}
.mono{font-family:'DM Mono',monospace}.eyebrow{color:var(--cyan);font-size:.62rem;font-weight:800;letter-spacing:.2em;text-transform:uppercase}.muted{color:var(--muted)}
.brand{padding:.1rem 0 1rem}.brand-row{display:flex;align-items:center;gap:.7rem}.brand-logo{width:42px;height:42px;border-radius:12px;background:white;padding:5px;object-fit:contain}.brand-mark{color:var(--cyan);font-weight:800;font-size:.72rem}.brand-title{font-size:1rem;font-weight:800}.brand-sub{font-size:.58rem;color:var(--muted);margin-top:.12rem}.nav-label{font-size:.58rem;color:#64748b;text-transform:uppercase;letter-spacing:.18em;font-weight:800;margin:1rem 0 .45rem}.team-mini{font-size:.59rem;color:#a3afbd;line-height:1.75;border-top:1px solid var(--line);padding-top:.75rem}.team-mini strong{color:#e5edf5}
.hero{position:relative;overflow:hidden;padding:2.9rem;border:1px solid var(--line);border-radius:30px;background:radial-gradient(circle at 91% 5%,rgba(34,211,238,.2),transparent 25%),radial-gradient(circle at 75% 115%,rgba(167,139,250,.15),transparent 32%),linear-gradient(135deg,#07111c,#06101a 56%,#0c1020);margin-bottom:1.25rem;box-shadow:0 25px 80px rgba(0,0,0,.25)}
.hero:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(to right,transparent 25%,black 70%);pointer-events:none}.hero:after{content:'';position:absolute;right:-100px;top:-160px;width:420px;height:420px;border:1px solid rgba(34,211,238,.14);border-radius:50%;box-shadow:0 0 0 35px rgba(34,211,238,.025),0 0 0 75px rgba(34,211,238,.012)}
.hero h1{position:relative;margin:.4rem 0 0;font-size:clamp(3rem,6vw,6rem);line-height:.86;letter-spacing:-.08em;font-weight:800}.hero h1 span{color:var(--cyan)}.hero p{position:relative;color:#aab8c8;max-width:900px;font-size:.9rem;line-height:1.8;margin:1rem 0 0}.hero-meta{position:relative;display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.4rem}.pill{border:1px solid var(--line);background:rgba(255,255,255,.035);padding:.45rem .72rem;border-radius:999px;color:#abb8c7;font-size:.61rem}.pill strong{color:#fff}
.page-title{font-size:2.5rem;font-weight:800;letter-spacing:-.07em;margin:.05rem 0 .2rem}.page-title span{color:var(--cyan)}.page-sub{color:var(--muted);font-size:.78rem;margin-bottom:1.25rem}.section-title{margin:1rem 0 .6rem;font-size:.61rem;text-transform:uppercase;letter-spacing:.18em;color:#8fa0b5;font-weight:800}
.card{border:1px solid var(--line);background:linear-gradient(145deg,#09111b,#070c14);border-radius:18px;padding:1.1rem;height:100%;box-shadow:0 12px 35px rgba(0,0,0,.14)}.card-k{color:#718198;font-size:.56rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800}.card-v{font-family:'DM Mono',monospace;font-size:1.55rem;margin-top:.3rem}.card-note{color:var(--muted);font-size:.63rem;margin-top:.3rem;line-height:1.5}
label,.stNumberInput label,.stSelectbox label,.stFileUploader label{color:#b7c1ce!important;font-size:.67rem!important;font-weight:600!important}input,[data-baseweb="select"]>div{background:#070d16!important;border-color:#1a2a3d!important;color:#f5f7fb!important;border-radius:10px!important}.stButton>button{border:1px solid rgba(34,211,238,.38);background:linear-gradient(135deg,#0c202a,#09151e);color:#e6fbff;border-radius:11px;min-height:43px;font-weight:800;transition:.2s}.stButton>button:hover{border-color:var(--cyan);color:#fff;transform:translateY(-1px);box-shadow:0 8px 25px rgba(34,211,238,.12)}.stDownloadButton>button{border-radius:11px}
.range-note{font-size:.56rem;color:#66768b;margin-top:-.35rem;margin-bottom:.55rem}.decision{min-height:305px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border:1px solid var(--line);border-radius:22px;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.12),transparent 48%),var(--panel);padding:1.5rem;box-shadow:0 20px 50px rgba(0,0,0,.2)}.decision .status{font-size:.58rem;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:800}.decision .label{margin-top:.45rem;font-size:2.8rem;line-height:1;font-weight:800;letter-spacing:-.06em}.approved{color:var(--lime)}.rejected{color:var(--red)}.prob{font-family:'DM Mono',monospace;margin-top:.8rem;font-size:1.05rem}.confidence-bar{width:82%;height:7px;background:#172232;border-radius:99px;overflow:hidden;margin-top:1rem}.confidence-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));border-radius:99px}.decision-note{color:var(--muted);font-size:.61rem;margin-top:.7rem}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:.7rem}.metric{border:1px solid var(--line);background:#090f19;border-radius:13px;padding:.8rem}.metric .k{color:var(--muted);font-size:.55rem;text-transform:uppercase;letter-spacing:.1em}.metric .v{font-family:'DM Mono',monospace;font-size:1rem;margin-top:.2rem}.risk{padding:.65rem .75rem;border-left:2px solid var(--lime);background:rgba(163,230,53,.035);margin:.35rem 0;border-radius:0 9px 9px 0;color:#aeb9c8;font-size:.65rem;line-height:1.45}.risk.warn{border-left-color:var(--amber)}
.feature-chip{display:inline-block;border:1px solid #1d3348;background:#09121c;border-radius:999px;padding:.35rem .55rem;margin:.18rem;color:#b7c5d5;font-family:'DM Mono',monospace;font-size:.56rem}.feature-chip b{color:var(--cyan)}.insight{border:1px solid var(--line);border-radius:16px;padding:1rem;background:linear-gradient(145deg,#09121c,#070d15);margin-bottom:.7rem}.insight h4{margin:0;font-size:.75rem}.insight p{margin:.3rem 0 0;color:var(--muted);font-size:.65rem;line-height:1.6}.about-hero{border:1px solid var(--line);border-radius:24px;padding:2rem;background:radial-gradient(circle at 90% 10%,rgba(167,139,250,.13),transparent 30%),#080e17}.team-card{border:1px solid var(--line);border-radius:16px;background:#090f18;padding:1rem;height:100%}.team-number{font-family:'DM Mono',monospace;color:var(--cyan);font-size:.62rem}.team-name{font-size:.79rem;font-weight:700;margin-top:.35rem}.team-role{color:var(--muted);font-size:.6rem;margin-top:.25rem}.footer-note{text-align:center;color:#536176;font-size:.56rem;margin-top:2rem;letter-spacing:.06em}
.present-tip{display:flex;justify-content:space-between;align-items:center;gap:1rem;border:1px solid var(--line);background:#07101a;padding:.65rem .8rem;border-radius:12px;margin-bottom:.65rem;color:#8191a5;font-size:.61rem}.present-tip b{color:#dce8f2}
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

with st.spinner("Booting Loan Intelligence…"):
    bundle, metrics = get_model()
reference = get_reference_data()

BENCHMARK = pd.DataFrame([
    ["Logistic Regression",86.55,64,92,75],["KNN",86.29,64,88,74],["Decision Tree",89.28,73,82,77],
    ["Random Forest",89.48,71,88,79],["SVM",88.02,67,92,77],["XGBoost",92.87,87,80,83]
],columns=["Model","Accuracy","Precision","Recall","F1"])
RANGES={
    "age":(18,80),"income":(float(reference.person_income.min()),float(reference.person_income.max())),
    "emp_exp":(0,60),"loan_amount":(float(reference.loan_amnt.min()),float(reference.loan_amnt.max())),
    "interest":(float(reference.loan_int_rate.min()),float(reference.loan_int_rate.max())),
    "loan_percent_income":(float(reference.loan_percent_income.min()),min(1.0,float(reference.loan_percent_income.max()))),
    "credit_history":(float(reference.cb_person_cred_hist_length.min()),float(reference.cb_person_cred_hist_length.max())),
    "credit_score":(int(reference.credit_score.min()),int(reference.credit_score.max()))}

def clamp(v,lo,hi): return max(lo,min(hi,v))
def card(k,v,note=""): return f'<div class="card"><div class="card-k">{html.escape(str(k))}</div><div class="card-v">{html.escape(str(v))}</div><div class="card-note">{html.escape(str(note))}</div></div>'


def presentation_html():
    slides=[
      {"k":"01 / OPENING","title":"Loan <span>Intelligence.</span>","sub":"An end-to-end Machine Learning system for binary loan-status prediction.","body":"NTI Machine Learning track final project connecting data preparation, class-imbalance handling, model benchmarking, XGBoost inference, and an interactive product experience.","stat":"92.87%","label":"held-out test accuracy","chips":["MACHINE LEARNING","BINARY CLASSIFICATION","XGBOOST","NTI"]},
      {"k":"02 / PROBLEM","title":"Turn applicant signals into a <span>repeatable decision.</span>","sub":"Loan status is framed as a binary classification problem.","body":"The system receives applicant and loan attributes, applies the same preprocessing path used by the project workflow, and returns a predicted class plus probability estimates. The focus is reproducibility: the result should come from a defined pipeline rather than an unexplained score.","quote":"A strong demo is not only a prediction. It is a traceable ML workflow that can be inspected, tested, and defended.","chips":["INPUT → TRANSFORM → PREDICT","REPEATABLE PIPELINE"]},
      {"k":"03 / DATASET","title":"45K+ rows. <span>13 raw inputs.</span>","sub":"The supplied dataset combines applicant, employment, loan, credit, and target information.","stat":f"{len(reference):,}","label":"source rows","body":"The raw table includes demographic attributes, income and employment experience, home ownership, loan intent, loan amount, interest rate, loan-to-income ratio, credit history, credit score, previous default history, and loan status.","chips":["NUMERIC + CATEGORICAL","13 RAW MODEL INPUTS","LOAN_STATUS TARGET"]},
      {"k":"04 / DATA QUALITY","title":"Clean the signal. <span>Expose the assumptions.</span>","sub":"EDA inspects structure, duplicates, missingness, distributions, correlations, and categorical counts.","body":"The workflow removes loan_id from modeling, applies IQR-based clipping to selected numeric variables, and keeps the age and employment-experience constraints used in the project. This creates a controlled feature table before encoding and scaling.","chips":["EDA","IQR CLIPPING","BOUNDS","QUALITY CHECKS"]},
      {"k":"05 / PREPROCESSING","title":"From raw records to <span>model-ready features.</span>","sub":"Categorical information is transformed into numerical representations.","body":"Gender and previous-default history are binary mapped. Education is ordinally encoded from High School through Doctorate. Home ownership and loan intent are one-hot encoded with drop_first=True. RobustScaler then transforms the feature space before model fitting.","chips":["BINARY MAP","ORDINAL MAP","ONE-HOT","ROBUSTSCALER"]},
      {"k":"06 / IMBALANCE","title":"The training target was <span>imbalanced.</span>","sub":"Class 0 started with substantially more training examples than class 1.","stat":"27,887 × 2","label":"balanced rows after SMOTETomek","body":"SMOTETomek is applied only to the training data after scaling. SMOTE synthesizes minority examples while Tomek links clean borderline pairs. The held-out test set remains untouched for evaluation.","chips":["SMOTE","TOMEK LINKS","TRAINING ONLY"]},
      {"k":"07 / MODEL ARENA","title":"Six classifiers entered the <span>benchmark.</span>","sub":"Model selection is evidence-driven.","body":"Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost were compared on the same held-out test split. The benchmark exposes both overall accuracy and class-1 precision, recall, and F1 so that one metric does not hide the trade-offs.","chips":["LOGISTIC","KNN","TREE","RANDOM FOREST","SVM","XGBOOST"]},
      {"k":"08 / WINNER","title":"XGBoost led the benchmark at <span>92.87%.</span>","sub":"Strongest overall held-out accuracy, class-1 precision, and class-1 F1 in this benchmark.","stat":"92.87%","label":"XGBoost test accuracy","body":"XGBoost reached 0.87 precision, 0.80 recall, and 0.83 F1 for class 1. Logistic Regression and SVM reached higher class-1 recall at 0.92, so the selection is a trade-off rather than a claim of universal superiority.","chips":["PRECISION 87%","RECALL 80%","F1 83%"]},
      {"k":"09 / GENERALIZATION","title":"A high training score is not the <span>whole story.</span>","sub":"Held-out performance is the anchor for the product.","stat":"97.68 → 92.87","label":"XGBoost train → test accuracy","body":"The Decision Tree reaches 100% training accuracy but a lower test score, a clear overfitting signal. XGBoost also has a train/test gap, so further validation and tuning would be appropriate before production use.","chips":["HELD-OUT TEST","OVERFITTING AWARENESS"]},
      {"k":"10 / ARCHITECTURE","title":"The notebook became an <span>interactive product.</span>","sub":"Reusable model logic powers every inference surface.","body":"Reference CSV → preprocessing and encoding → RobustScaler → XGBoost → class + probability. SMOTETomek is part of training only. The Streamlit layer adds single-record prediction, batch scoring, diagnostics, documentation, and this presentation.","chips":["PYTHON","PANDAS","SCIKIT-LEARN","IMBLEARN","XGBOOST","STREAMLIT"]},
      {"k":"11 / PREDICTION STUDIO","title":"Enter an applicant. <span>See the decision.</span>","sub":"All 13 raw model inputs are exposed with bounded numeric controls.","body":"The Prediction Studio validates numeric ranges, constructs the raw applicant record, applies the trained bundle's feature layout, and displays predicted status, approval probability, rejection probability, and benchmark context.","chips":["13 INPUTS","RANGE-GUARDED","LIVE INFERENCE"]},
      {"k":"12 / BATCH LAB","title":"One applicant or <span>an entire CSV.</span>","sub":"Batch scoring turns the demo into a reusable inference utility.","body":"Users can download a schema template, upload raw applicant rows, validate required columns, score the file with the XGBoost bundle, inspect approved/rejected counts, and download the scored CSV. If loan_status is supplied, row-level correctness is also calculated.","chips":["UPLOAD","VALIDATE","SCORE","DOWNLOAD"]},
      {"k":"13 / MODEL INSIGHTS","title":"Don't just show the answer. <span>Show the evidence.</span>","sub":"Diagnostics make the final model choice defendable.","body":"The Model Insights workspace exposes the six-model benchmark, XGBoost feature importance, confusion-matrix values, and the precision/recall trade-off. This keeps the product presentation connected to the actual experiment rather than a decorative dashboard.","chips":["BENCHMARK","FEATURE IMPORTANCE","CONFUSION MATRIX"]},
      {"k":"14 / LIMITATIONS","title":"Good ML work includes <span>what could be better.</span>","sub":"This is an educational prototype, not a production credit-decision engine.","body":"Results are tied to the supplied dataset and split. The current workflow computes IQR bounds before the split, which can introduce mild leakage. Education is ordinal-encoded, which imposes an ordering assumption. Probability outputs are not presented as calibrated financial risk.","chips":["DATASET-SPECIFIC","CALIBRATION NEEDED","FAIRNESS REVIEW","LEAKAGE HARDENING"]},
      {"k":"15 / ROADMAP","title":"From student project to <span>ML service.</span>","sub":"The next gains come from stronger validation and engineering.","body":"Next: fit preprocessing statistics strictly inside training folds, tune XGBoost with validation, calibrate probabilities, add cross-validation and threshold analysis, version data and models, monitor drift, and expose the model through a controlled API with audit logging.","chips":["CV","CALIBRATION","MLOPS","MONITORING","API"]},
      {"k":"16 / CLOSING","title":"Four people. <span>One ML system.</span>","sub":"NTI Machine Learning Track — Loan Status Prediction.","body":"The final experience combines analysis, preprocessing, benchmarking, inference, batch scoring, diagnostics, documentation, and presentation into one coherent project.","chips":TEAM}
    ]
    data=json.dumps(slides,ensure_ascii=False).replace("</","<\\/")
    return f"""<!doctype html><html><head><meta charset='utf-8'><link href='https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap' rel='stylesheet'><style>
*{{box-sizing:border-box}}html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#03060b;color:#f7fafc;font-family:Manrope,Arial,sans-serif}}#stage{{width:100%;height:100vh;min-height:650px;position:relative;overflow:hidden;background:radial-gradient(circle at 88% 8%,rgba(34,211,238,.13),transparent 25%),radial-gradient(circle at 72% 110%,rgba(167,139,250,.11),transparent 30%),#03060b}}#grid{{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:38px 38px;mask-image:linear-gradient(to bottom,black,transparent)}}#orb{{position:absolute;right:-130px;top:10%;width:460px;height:460px;border:1px solid rgba(34,211,238,.11);border-radius:50%;box-shadow:0 0 0 45px rgba(34,211,238,.018),0 0 0 90px rgba(34,211,238,.01)}}#top{{position:absolute;left:5%;right:5%;top:4%;display:flex;justify-content:space-between;align-items:center;z-index:5}}.brand{{font:500 10px 'DM Mono',monospace;letter-spacing:.16em;color:#7890a8}}.brand b{{color:#22d3ee}}#counter{{font:500 10px 'DM Mono',monospace;color:#6e8196}}#slide{{position:absolute;inset:11% 5% 11%;display:flex;flex-direction:column;justify-content:center;z-index:2;animation:enter .42s ease}}@keyframes enter{{from{{opacity:0;transform:translateY(15px)}}to{{opacity:1;transform:none}}}}.kicker{{font:800 10px 'DM Mono',monospace;letter-spacing:.19em;color:#22d3ee;text-transform:uppercase}}h1{{font-size:clamp(42px,6.1vw,92px);line-height:.91;letter-spacing:-.075em;max-width:1120px;margin:14px 0 16px;font-weight:800}}h1 span{{color:#22d3ee}}.sub{{font-size:clamp(16px,1.6vw,23px);line-height:1.45;color:#d8e1eb;max-width:930px}}.body{{font-size:14px;line-height:1.8;color:#94a5b8;max-width:900px;margin-top:18px}}.stat{{font:500 clamp(54px,8vw,112px) 'DM Mono',monospace;color:#22d3ee;letter-spacing:-.08em;margin-top:20px;line-height:.95}}.statlabel{{font:800 10px 'DM Mono',monospace;color:#61748a;text-transform:uppercase;letter-spacing:.15em;margin-top:8px}}.chips{{display:flex;gap:7px;flex-wrap:wrap;margin-top:24px;max-width:1100px}}.chip{{border:1px solid #203449;background:rgba(255,255,255,.035);border-radius:999px;padding:7px 10px;color:#aebdcc;font:500 9px 'DM Mono',monospace}}.quote{{border-left:2px solid #22d3ee;padding-left:16px;color:#d3dee8;max-width:820px;margin-top:22px;font-size:16px;line-height:1.7}}#bottom{{position:absolute;left:5%;right:5%;bottom:4%;z-index:5;display:flex;align-items:center;gap:10px}}#progress{{height:2px;background:#152233;flex:1;border-radius:10px;overflow:hidden}}#bar{{height:100%;background:linear-gradient(90deg,#22d3ee,#a3e635);width:0;transition:.35s}}button{{border:1px solid #23374b;background:#08111b;color:#c9d7e4;border-radius:9px;padding:8px 12px;font:800 9px 'DM Mono',monospace;cursor:pointer}}button:hover{{border-color:#22d3ee;color:#fff}}#hint{{color:#53667b;font:500 8px 'DM Mono',monospace}}.full{{position:absolute;right:5%;top:10%;z-index:10}}
</style></head><body><div id='stage'><div id='grid'></div><div id='orb'></div><div id='top'><div class='brand'><b>NTI</b> / MACHINE LEARNING / LOAN INTELLIGENCE</div><div id='counter'>01 / 16</div></div><button class='full' onclick='full()'>⛶ FULLSCREEN</button><div id='slide'></div><div id='bottom'><button onclick='prev()'>← PREV</button><button onclick='next()'>NEXT →</button><div id='progress'><div id='bar'></div></div><div id='hint'>← → · SPACE · F · DOUBLE CLICK</div></div></div><script>const slides={data};let i=0;const root=document.getElementById('slide');function render(){{const s=slides[i];root.style.animation='none';void root.offsetWidth;root.style.animation='enter .42s ease';let h=`<div class='kicker'>${{s.k}}</div><h1>${{s.title}}</h1><div class='sub'>${{s.sub}}</div>`;if(s.stat)h+=`<div class='stat'>${{s.stat}}</div><div class='statlabel'>${{s.label}}</div>`;h+=`<div class='body'>${{s.body}}</div>`;if(s.quote)h+=`<div class='quote'>${{s.quote}}</div>`;h+=`<div class='chips'>${{(s.chips||[]).map(x=>`<div class='chip'>${{x}}</div>`).join('')}}</div>`;root.innerHTML=h;document.getElementById('counter').textContent=String(i+1).padStart(2,'0')+' / '+String(slides.length).padStart(2,'0');document.getElementById('bar').style.width=((i+1)/slides.length*100)+'%'}}function next(){{i=Math.min(slides.length-1,i+1);render()}}function prev(){{i=Math.max(0,i-1);render()}}function full(){{const e=document.getElementById('stage');if(!document.fullscreenElement)e.requestFullscreen?.();else document.exitFullscreen?.()}}document.addEventListener('keydown',e=>{{if(['ArrowRight','PageDown',' '].includes(e.key)){{e.preventDefault();next()}}else if(['ArrowLeft','PageUp'].includes(e.key)){{e.preventDefault();prev()}}else if(e.key.toLowerCase()==='f')full()}});document.getElementById('stage').addEventListener('dblclick',full);render();</script></body></html>"""

with st.sidebar:
    st.markdown(f'<div class="brand"><div class="brand-row"><img class="brand-logo" src="{NTI_LOGO}"><div><div class="brand-mark">NTI / ML</div><div class="brand-title">Loan Intelligence</div><div class="brand-sub">Interactive classification system</div></div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Workspace</div>',unsafe_allow_html=True)
    page=st.radio("Navigate",["Prediction Studio","Batch Lab","Project Presentation","Model Insights","Team & About"],label_visibility="collapsed")
    st.markdown('<div class="nav-label">Live system</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="risk"><strong>XGBoost</strong><br>{metrics["accuracy"]:.2%} held-out test accuracy</div><div class="risk"><strong>{len(reference):,} source rows</strong><br>{len(bundle.feature_columns)} encoded features</div><div class="risk"><strong>SMOTETomek</strong><br>{metrics["train_rows_after_smotetomek"]:,} balanced training rows</div>',unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Team</div>',unsafe_allow_html=True)
    st.markdown('<div class="team-mini">'+'<br>'.join(f'<strong>{i+1}.</strong> {html.escape(n)}' for i,n in enumerate(TEAM))+'</div>',unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:.8rem;font-size:.58rem"><a href="{NTI_SITE}" target="_blank" style="color:#22d3ee;text-decoration:none">Official NTI website ↗</a></div>',unsafe_allow_html=True)

if page=="Prediction Studio":
    st.markdown(f'<div class="hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK / FINAL PROJECT</div><h1>Loan <span>Intelligence.</span></h1><p>Interactive binary loan-status prediction using the reusable XGBoost workflow. Numeric controls are bounded by the reference dataset, while inference uses the model bundle in <span class="mono">src/model.py</span>.</p><div class="hero-meta"><div class="pill"><strong>{metrics["accuracy"]:.2%}</strong> test accuracy</div><div class="pill"><strong>{len(reference):,}</strong> source rows</div><div class="pill"><strong>{len(bundle.feature_columns)}</strong> encoded features</div><div class="pill"><strong>6</strong> classifiers benchmarked</div></div></div>',unsafe_allow_html=True)
    left,right=st.columns([1.18,.82],gap='large')
    with left:
        st.markdown('<div class="section-title">01 / Applicant profile — all 13 model inputs</div>',unsafe_allow_html=True)
        with st.container(border=True):
            a,b=st.columns(2)
            with a:
                age=st.number_input('Age',min_value=18,max_value=80,value=30,step=1)
                st.markdown('Dataset guardrail: 18 → 80',unsafe_allow_html=True)
                income=st.number_input('Annual income',min_value=RANGES['income'][0],max_value=RANGES['income'][1],value=clamp(60000,*RANGES['income']),step=1000.0)
                st.markdown(f'<div class="range-note">Dataset: {RANGES["income"][0]:,.0f} → {RANGES["income"][1]:,.0f}</div>',unsafe_allow_html=True)
                emp=st.number_input('Employment experience (years)',min_value=0,max_value=60,value=5,step=1)
                edu=st.selectbox('Education',['High School','Associate','Bachelor','Master','Doctorate'],index=2)
                gender=st.selectbox('Gender',['male','female'])
                home=st.selectbox('Home ownership',['RENT','OWN','MORTGAGE','OTHER'])
                default=st.selectbox('Previous loan default',['No','Yes'])
            with b:
                intent=st.selectbox('Loan intent',['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT'])
                amount=st.number_input('Loan amount',min_value=RANGES['loan_amount'][0],max_value=RANGES['loan_amount'][1],value=clamp(10000,*RANGES['loan_amount']),step=500.0)
                rate=st.number_input('Interest rate (%)',min_value=RANGES['interest'][0],max_value=RANGES['interest'][1],value=clamp(10,*RANGES['interest']),step=.1)
                ratio=st.number_input('Loan / income ratio',min_value=RANGES['loan_percent_income'][0],max_value=RANGES['loan_percent_income'][1],value=clamp(.2,*RANGES['loan_percent_income']),step=.01)
                history=st.number_input('Credit history length (years)',min_value=RANGES['credit_history'][0],max_value=RANGES['credit_history'][1],value=clamp(4,*RANGES['credit_history']),step=.5)
                score=st.number_input('Credit score',min_value=RANGES['credit_score'][0],max_value=RANGES['credit_score'][1],value=clamp(680,*RANGES['credit_score']),step=1)
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
        else: st.markdown('<div class="decision"><div class="status">READY FOR INFERENCE</div><div class="label" style="color:#c7d2df">AWAITING INPUT</div><div class="prob">Fill the profile and run the model.</div><div class="confidence-bar"><div class="confidence-fill" style="width:0%"></div></div><div class="decision-note">The exact trained feature layout will be used.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">03 / System pulse</div>',unsafe_allow_html=True)
    x,y,z=st.columns(3)
    x.markdown(card('Best benchmark','XGBoost','Highest test accuracy in the six-model comparison'),unsafe_allow_html=True)
    y.markdown(card('Class-1 F1','0.83','XGBoost on held-out test data'),unsafe_allow_html=True)
    z.markdown(card('Training balance','SMOTETomek','Applied only to training data'),unsafe_allow_html=True)

elif page=="Batch Lab":
    st.markdown('<div class="page-title">Batch <span>Lab.</span></div><div class="page-sub">Upload raw applicant records, validate the schema, score them, and export the results.</div>',unsafe_allow_html=True)
    template_cols=MODEL_INPUT_COLUMNS+['loan_id','loan_status']; template=pd.DataFrame([{c:'' for c in template_cols}]); template.loc[0,'person_age']=30;template.loc[0,'person_income']=60000;template.loc[0,'person_home_ownership']='RENT';template.loc[0,'person_emp_exp']=5;template.loc[0,'loan_intent']='EDUCATION';template.loc[0,'loan_amnt']=10000;template.loc[0,'loan_int_rate']=10;template.loc[0,'loan_percent_income']=.2;template.loc[0,'cb_person_cred_hist_length']=4;template.loc[0,'credit_score']=680;template.loc[0,'previous_loan_defaults_on_file']='No';template.loc[0,'person_gender']='male';template.loc[0,'person_education']='Bachelor';template.loc[0,'loan_id']='example_001'
    l,r=st.columns([.75,1.25],gap='large')
    with l:
        st.markdown('<div class="section-title">01 / Schema</div>',unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-k">Required raw fields</div><div style="margin-top:.5rem">'+''.join(f'<span class="feature-chip"><b>•</b> {html.escape(c)}</span>' for c in MODEL_INPUT_COLUMNS)+'</div><div class="card-note">loan_id is optional. loan_status is optional and only used for evaluation when present.</div></div>',unsafe_allow_html=True)
        st.download_button('DOWNLOAD CSV TEMPLATE',template.to_csv(index=False).encode(),file_name='loan_prediction_template.csv',mime='text/csv',use_container_width=True)
        upload=st.file_uploader('Upload applicant CSV',type=['csv'])
    with r:
        if upload is None: st.markdown('<div class="about-hero"><div class="eyebrow mono">BATCH INFERENCE</div><h2 style="font-size:2rem;letter-spacing:-.05em">Score many applicants in one run.</h2><p class="muted">Upload the template or a CSV containing the 13 required raw model inputs.</p><div class="risk">The schema is validated before inference. Missing columns are rejected instead of silently producing partial predictions.</div></div>',unsafe_allow_html=True)
        else:
            try:
                df=pd.read_csv(upload); missing=[c for c in MODEL_INPUT_COLUMNS if c not in df.columns]
                if missing: st.error('Missing required columns: '+', '.join(missing))
                else:
                    st.success(f'Validated {len(df):,} rows.')
                    st.dataframe(df.head(8),use_container_width=True,hide_index=True)
                    if st.button('RUN BATCH INFERENCE →',use_container_width=True,type='primary'):
                        try: st.session_state.batch_scored=predict_batch(bundle,df);st.success(f'Scored {len(st.session_state.batch_scored):,} rows.')
                        except Exception as e: st.error(f'Inference failed: {e}')
            except Exception as e: st.error(f'Could not read CSV: {e}')
    scored=st.session_state.get('batch_scored')
    if scored is not None:
        st.markdown('<div class="section-title">02 / Results</div>',unsafe_allow_html=True)
        a,b,c=st.columns(3);a.markdown(card('Approved',int((scored.prediction==1).sum()),'Predicted rows'),unsafe_allow_html=True);b.markdown(card('Rejected',int((scored.prediction==0).sum()),'Predicted rows'),unsafe_allow_html=True);c.markdown(card('Rows scored',len(scored),'Inference output'),unsafe_allow_html=True)
        if 'actual_status' in scored.columns: st.info(f'Batch accuracy against supplied loan_status: {float(scored.correct.mean()):.2%}')
        st.dataframe(scored,use_container_width=True,hide_index=True)
        st.download_button('DOWNLOAD SCORED CSV',scored.to_csv(index=False).encode(),file_name='loan_predictions_scored.csv',mime='text/csv',use_container_width=True)

elif page=="Project Presentation":
    st.markdown('<div class="page-title">Project <span>Presentation.</span></div><div class="page-sub">A presentation mode inside the product — animated, keyboard-driven, and designed for fullscreen delivery.</div>',unsafe_allow_html=True)
    st.markdown('<div class="present-tip"><span><b>← →</b> navigate · <b>SPACE</b> next · <b>F</b> fullscreen · <b>double-click</b> fullscreen</span><span class="mono">16 SLIDES / FULL STORY</span></div>',unsafe_allow_html=True)
    components.html(presentation_html(),height=770,scrolling=False)
    st.caption('Use FULLSCREEN inside the presentation for the cleanest projector/demo experience.')

elif page=="Model Insights":
    st.markdown('<div class="page-title">Model <span>Insights.</span></div><div class="page-sub">The evidence behind the final model choice.</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4);a.markdown(card('Accuracy','92.87%','XGBoost held-out test'),unsafe_allow_html=True);b.markdown(card('Class-1 precision','87%','XGBoost'),unsafe_allow_html=True);c.markdown(card('Class-1 recall','80%','XGBoost'),unsafe_allow_html=True);d.markdown(card('Class-1 F1','83%','XGBoost'),unsafe_allow_html=True)
    st.markdown('<div class="section-title">Six-model benchmark</div>',unsafe_allow_html=True);show=BENCHMARK.copy();show['Accuracy']=show['Accuracy'].map(lambda x:f'{x:.2f}%');
    for col in ['Precision','Recall','F1']: show[col]=show[col].map(lambda x:f'{x:.0f}%')
    st.dataframe(show,use_container_width=True,hide_index=True)
    l,r=st.columns(2,gap='large')
    with l:
        st.markdown('<div class="section-title">XGBoost feature importance</div>',unsafe_allow_html=True)
        imp=pd.Series(bundle.model.feature_importances_,index=bundle.feature_columns).sort_values(ascending=False).head(12)
        st.bar_chart(pd.DataFrame({'Importance':imp}))
    with r:
        st.markdown('<div class="section-title">Held-out confusion matrix</div>',unsafe_allow_html=True)
        cm=metrics['confusion_matrix'];st.markdown(f'<div class="card"><div class="card-k">Predicted × Actual</div><div style="font-family:DM Mono,monospace;font-size:1.35rem;line-height:2;margin-top:1rem">TN <span style="color:#22d3ee">{cm[0][0]:,}</span> · FP <span style="color:#fb7185">{cm[0][1]:,}</span><br>FN <span style="color:#fbbf24">{cm[1][0]:,}</span> · TP <span style="color:#a3e635">{cm[1][1]:,}</span></div><div class="card-note">Counts from the XGBoost held-out test set.</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="insight"><h4>Precision / recall trade-off</h4><p>XGBoost has stronger class-1 precision and F1, while Logistic Regression and SVM have higher class-1 recall. The final selection reflects the benchmark objective used here.</p></div>',unsafe_allow_html=True)
        st.markdown('<div class="insight"><h4>Generalization</h4><p>The Decision Tree reaches 100% training accuracy and a lower test result. XGBoost also has a train/test gap, so further validation remains important.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Encoded feature space</div>',unsafe_allow_html=True);st.markdown(''.join(f'<span class="feature-chip">{html.escape(c)}</span>' for c in bundle.feature_columns),unsafe_allow_html=True)

else:
    st.markdown('<div class="about-hero"><div class="eyebrow mono">NTI / MACHINE LEARNING TRACK</div><h1 style="font-size:3.3rem;letter-spacing:-.07em;margin:.5rem 0">Loan Intelligence <span style="color:#22d3ee">Project.</span></h1><p style="color:#aab8c8;max-width:850px;line-height:1.8">A complete educational ML product combining exploratory analysis, preprocessing, imbalance handling, model benchmarking, XGBoost inference, batch scoring, diagnostics, and a presentation layer.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Team</div>',unsafe_allow_html=True);cols=st.columns(4);roles=['ML workflow & integration','Data analysis & preprocessing','Model evaluation & experimentation','Application & presentation']
    for i,(col,name,role) in enumerate(zip(cols,TEAM,roles)): col.markdown(f'<div class="team-card"><div class="team-number">0{i+1}</div><div class="team-name">{html.escape(name)}</div><div class="team-role">{role}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">System at a glance</div>',unsafe_allow_html=True);cols=st.columns(4)
    for col,k,v,n in zip(cols,['DATA','FEATURES','MODELS','WINNER'],[f'{len(reference):,}',str(len(bundle.feature_columns)),'6','92.87%'],['source rows','encoded features','benchmarked classifiers','XGBoost test accuracy']): col.markdown(card(k,v,n),unsafe_allow_html=True)
    st.markdown('<div class="section-title">Technical stack</div>',unsafe_allow_html=True);st.markdown(''.join(f'<span class="feature-chip">{x}</span>' for x in ['Python','Pandas','Scikit-learn','imbalanced-learn','XGBoost','Streamlit','GitHub Actions']),unsafe_allow_html=True)
    st.markdown(f'<div class="footer-note">Educational ML prototype · <a href="{NTI_SITE}" target="_blank" style="color:#22d3ee">Official NTI website</a></div>',unsafe_allow_html=True)
