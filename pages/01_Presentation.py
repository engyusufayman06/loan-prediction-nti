from pathlib import Path
import html

import pandas as pd
import streamlit as st

from src.model import predict_one, train_model

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "loan_data.csv"

st.set_page_config(page_title="Loan Intelligence — Presentation", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@500;600;700;800&display=swap');
:root{--bg:#02050a;--panel:#07101a;--line:#173047;--text:#f7fbff;--muted:#8da0b5;--cyan:#22d3ee;--lime:#a3e635;--purple:#a78bfa;--red:#fb7185;--amber:#fbbf24}
html,body,[class*="css"]{font-family:Manrope,sans-serif}.stApp{background:radial-gradient(circle at 80% 0%,rgba(34,211,238,.07),transparent 30%),var(--bg);color:var(--text)}
.block-container{max-width:1500px;padding:1rem 2rem 3rem}.stButton>button{border:1px solid var(--line);background:#07111b;color:#eafaff;border-radius:12px;min-height:44px;font-weight:800}.stButton>button:hover{border-color:var(--cyan);box-shadow:0 0 28px rgba(34,211,238,.1);transform:translateY(-1px)}
[data-testid="stSidebar"]{background:#040911;border-right:1px solid var(--line)}
.pbar{height:5px;background:#0c1927;border-radius:99px;overflow:hidden;margin:0 0 1.1rem}.pfill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple),var(--lime));transition:width .7s ease}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem}.kicker{font:500 .62rem 'DM Mono';letter-spacing:.18em;color:var(--cyan)}.counter{font:500 .65rem 'DM Mono';color:#71849a}
.stage{position:relative;min-height:590px;border:1px solid var(--line);border-radius:30px;overflow:hidden;padding:3.4rem;background:radial-gradient(circle at 88% 10%,rgba(34,211,238,.16),transparent 25%),radial-gradient(circle at 15% 100%,rgba(167,139,250,.10),transparent 30%),linear-gradient(135deg,#07121d,#050b13 65%,#0b1020);box-shadow:0 30px 100px rgba(0,0,0,.35);animation:stageIn .55s ease both}
.stage:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(90deg,transparent,black 25%,black 75%,transparent);pointer-events:none}.stage:after{content:'';position:absolute;width:430px;height:430px;right:-180px;bottom:-230px;border:1px solid rgba(34,211,238,.16);border-radius:50%;box-shadow:0 0 0 45px rgba(34,211,238,.025),0 0 0 90px rgba(34,211,238,.015)}
@keyframes stageIn{from{opacity:0;transform:translateY(12px) scale(.99)}to{opacity:1;transform:none}}@keyframes pulse{0%,100%{opacity:.45}50%{opacity:1}}@keyframes scan{from{transform:translateY(-100%)}to{transform:translateY(600%)}}
.pause *{animation-play-state:paused!important}
.stage h1{position:relative;font-size:clamp(3rem,6vw,6.5rem);line-height:.88;letter-spacing:-.085em;margin:0 0 1rem;font-weight:800}.stage h1 span{color:var(--cyan)}.stage h2{position:relative;font-size:clamp(2rem,4vw,4rem);line-height:.95;letter-spacing:-.07em;margin:0 0 1rem}.stage h2 span{color:var(--cyan)}.sub{position:relative;color:#b3c0cf;max-width:900px;font-size:.92rem;line-height:1.8}.body{position:relative;color:#9aaabd;max-width:950px;font-size:.78rem;line-height:1.85;margin-top:1rem}.quote{position:relative;margin-top:1.2rem;padding:1rem 1.2rem;border-left:2px solid var(--cyan);background:rgba(34,211,238,.035);max-width:900px;color:#d4e3ef;font-size:.75rem;line-height:1.7}
.stats{position:relative;display:flex;gap:.7rem;flex-wrap:wrap;margin-top:1.7rem}.stat{border:1px solid var(--line);background:rgba(255,255,255,.035);border-radius:15px;padding:.85rem 1rem;min-width:145px}.stat .v{font:500 1.35rem 'DM Mono';color:#fff}.stat .l{font-size:.55rem;color:var(--muted);text-transform:uppercase;letter-spacing:.11em;margin-top:.2rem}.chips{position:relative;margin-top:1.3rem}.chip{display:inline-block;border:1px solid #1b3850;background:#08131e;border-radius:99px;padding:.4rem .65rem;margin:.18rem;color:#b9c8d6;font:500 .55rem 'DM Mono';letter-spacing:.04em}
.bars{position:relative;margin-top:1.5rem;max-width:1000px}.barrow{display:grid;grid-template-columns:180px 1fr 70px;gap:10px;align-items:center;margin:.55rem 0}.barname{font-size:.62rem;color:#b8c5d2}.track{height:11px;border-radius:99px;background:#0d1a28;overflow:hidden;border:1px solid #14283b}.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--purple));transform-origin:left;animation:grow 1s ease both}.fill.win{background:linear-gradient(90deg,var(--cyan),var(--lime))}@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}.barval{font:500 .65rem 'DM Mono';text-align:right;color:#eafaff}
.matrix{display:grid;grid-template-columns:repeat(2,150px);gap:8px;margin-top:1.4rem}.cell{padding:1.2rem;border:1px solid var(--line);background:#091522;border-radius:15px}.cell b{font:500 1.7rem 'DM Mono'}.cell small{display:block;color:var(--muted);font-size:.55rem;margin-top:.25rem}
.controls{display:flex;gap:.55rem;align-items:center;flex-wrap:wrap;margin:.8rem 0 1rem}.hint{font-size:.59rem;color:#687b90;margin-left:auto}.live{border:1px solid rgba(34,211,238,.3);border-radius:18px;background:rgba(34,211,238,.035);padding:1rem;margin-top:1.2rem}.live-title{font:500 .6rem 'DM Mono';letter-spacing:.16em;color:var(--cyan)}
.small-note{color:#667a91;font-size:.58rem;line-height:1.6;margin-top:.8rem}.stSelectbox label,.stNumberInput label,.stSlider label{font-size:.65rem!important;color:#aab8c8!important}.stNumberInput input,[data-baseweb="select"]>div{background:#07101a!important;border-color:#183047!important;color:#fff!important}
</style>
""", unsafe_allow_html=True)

if "slide" not in st.session_state: st.session_state.slide = 0
if "paused" not in st.session_state: st.session_state.paused = False
if "revealed" not in st.session_state: st.session_state.revealed = False

@st.cache_resource(show_spinner=False)
def get_model(): return train_model(DATA_PATH)
@st.cache_data(show_spinner=False)
def get_data(): return pd.read_csv(DATA_PATH)

bundle, metrics = get_model(); df = get_data()

slides = [
("01 / OPENING","Loan <span>Intelligence.</span>","End-to-end Machine Learning for loan-status classification.","A product-style ML project that connects research, preprocessing, imbalance handling, model benchmarking, XGBoost inference, diagnostics, and an interactive interface.","92.87%","held-out test accuracy",["NTI MACHINE LEARNING","BINARY CLASSIFICATION","XGBOOST"]),
("02 / THE QUESTION","Can historical applicant signals support a <span>repeatable classification</span> workflow?","The target is `loan_status`.","We frame the task as supervised binary classification: learn from labelled applicant and loan records, evaluate on held-out data, then reuse the same transformation contract during inference.","13","raw model inputs",["SUPERVISED LEARNING","CLASSIFICATION","REPRODUCIBILITY"]),
("03 / DATA","45K+ records. <span>13 inputs.</span>","The raw table mixes numeric and categorical applicant/loan attributes.","The project works with age, income, employment experience, ownership, intent, loan amount, interest rate, loan-to-income ratio, credit history, credit score, previous default history, gender, education, and the loan-status target.",f"{len(df):,}","source rows",["NUMERIC","CATEGORICAL","LOAN_STATUS"]),
("04 / EDA","Before modeling: <span>understand the data.</span>","The notebook performs structural and distributional exploration.","EDA covers shape, columns, info, descriptive statistics, duplicates, null checks, value counts, histograms, correlations, boxplots, scatterplots, categorical counts, target distribution, and target-related plots.","EDA","research stage",["DISTRIBUTIONS","CORRELATION","OUTLIERS","TARGET"]),
("05 / CLEANING","Control the extremes. <span>Keep the workflow explicit.</span>","The preprocessing contract removes identifiers and controls selected numeric extremes.","`loan_id` is removed. IQR clipping is applied to selected numeric columns. Age and employment experience are constrained to the project bounds before categorical encoding.","IQR","outlier control",["LOAN_ID REMOVED","IQR CLIPPING","SANITY FILTERS"]),
("06 / REPRESENTATION","Raw categories → <span>model-ready features.</span>","Different variable types receive different encodings.","Gender and previous-default history are binary mapped. Education is ordinally encoded. Home ownership and loan intent use one-hot encoding. RobustScaler then transforms the numerical feature space.","20","encoded features",["BINARY","ORDINAL","ONE-HOT","ROBUSTSCALER"]),
("07 / IMBALANCE","The learner saw a <span>skewed target.</span>","The original training distribution was 27,991 class-0 vs 8,001 class-1 examples.","SMOTETomek is applied to the scaled training split only. After resampling, the project reports 27,887 examples in each class. The held-out test set is kept untouched for evaluation.","1 : 1","post-resampling training balance",["SMOTE","TOMEK LINKS","TRAINING ONLY"]),
("08 / MODEL ARENA","Six models. <span>One benchmark.</span>","Same project split, comparable evaluation.","Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost were compared. The project reports accuracy plus class-1 precision, recall, and F1 to expose trade-offs.","6","classifiers benchmarked",["LOGISTIC","KNN","TREE","RANDOM FOREST","SVM","XGBOOST"]),
("09 / SCOREBOARD","XGBoost reaches <span>92.87%.</span>","It leads this benchmark on test accuracy, class-1 precision, and class-1 F1.","The selection is benchmark-specific—not a claim that XGBoost is universally superior. Logistic Regression and SVM have higher class-1 recall, which matters when missing positive cases is more costly.","0.83","class-1 F1",["92.87% ACCURACY","0.87 PRECISION","0.80 RECALL"]),
("10 / GENERALIZATION","The training score is not the story. <span>The test score is.</span>","Overfitting is explicitly visible in the comparison.","Decision Tree reaches 100% training accuracy but 89.28% test accuracy. XGBoost reaches 97.68% training accuracy and 92.87% test accuracy. A gap remains, so further validation is still required.","+","held-out thinking",["TRAIN ≠ TEST","OVERFITTING","GENERALIZATION"]),
("11 / ERROR PROFILE","What did the selected model <span>get wrong?</span>","Held-out XGBoost confusion matrix.","The matrix contains true negatives, false positives, false negatives, and true positives. Reading these counts is more informative than presenting accuracy alone.","2,?","see live matrix",["TN","FP","FN","TP"]),
("12 / LIVE LAB","Don't just watch it. <span>Run it.</span>","This slide is connected to the actual XGBoost model in the repository.","Change an applicant profile below and execute the real reusable inference function. The output is generated by the project's trained model bundle—not a presentation-only mock.","LIVE","model inference",["INTERACTIVE","REAL PIPELINE","PREDICT_ONE"]),
("13 / PRODUCT","From notebook → <span>ML product.</span>","The application exposes the model as an experience.","Prediction Studio handles individual applicants. Batch Lab handles CSV scoring and schema validation. Model Insights exposes benchmark evidence. The presentation itself becomes a guided technical interface.","03","core workspaces",["STUDIO","BATCH LAB","INSIGHTS"]),
("14 / TRANSPARENCY","A prediction is useful only when its <span>limits are visible.</span>","The project is educational—not a lender's underwriting engine.","The dataset may not represent a current lender population. Fairness, calibration, subgroup performance, drift, and real-world probability validity have not been established. Probabilities should not be interpreted as guaranteed approval odds.","NO","production claim",["LIMITATIONS","RESPONSIBLE USE","NO REAL FINANCIAL DECISION"]),
("15 / HARDENING","What would make this <span>production-grade?</span>","The roadmap continues beyond the classroom demo.","Version model artifacts, expose an API, add SHAP explanations, calibrate probabilities, run fairness/subgroup evaluation, containerize deployment, track experiments with MLflow, register models, and monitor drift and service health.","NEXT","engineering roadmap",["FASTAPI","SHAP","MLFLOW","MONITORING"]),
("16 / CLOSING","Research → Engineering → <span>Product.</span>","The final result is bigger than one accuracy number.","Loan Intelligence demonstrates a complete ML journey: understand the data, build a reproducible pipeline, compare models honestly, expose the trade-offs, package inference, and make the result interactive.","4","team members",["YUSUF AYMAN TOLBA","MOHAMED REDA HUSSEIN","ABDELRAHMAN MOHAMED AHMED","ABDELMONIEM IBRAHIM ABDELMONIEM"]),
]

n=len(slides); i=st.session_state.slide; meta,title,sub,body,sv,sl,chips=slides[i]

st.markdown(f'<div class="top"><div class="kicker">LOAN INTELLIGENCE / NTI ML</div><div class="counter">{i+1:02d} / {n:02d}</div></div><div class="pbar"><div class="pfill" style="width:{(i+1)/n*100:.2f}%"></div></div>',unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6=st.columns([1,1,1,1,1,1.4])
with c1:
    if st.button("← PREV", use_container_width=True): st.session_state.slide=(i-1)%n; st.session_state.revealed=False; st.rerun()
with c2:
    if st.button("NEXT →", use_container_width=True): st.session_state.slide=(i+1)%n; st.session_state.revealed=False; st.rerun()
with c3:
    if st.button("HOME", use_container_width=True): st.session_state.slide=0; st.session_state.revealed=False; st.rerun()
with c4:
    st.session_state.paused=st.toggle("PAUSE MOTION",value=st.session_state.paused)
with c5:
    if st.button("REVEAL DETAIL", use_container_width=True): st.session_state.revealed=not st.session_state.revealed
with c6:
    chosen=st.selectbox("JUMP",[f"{j+1:02d} · {slides[j][0].split(' / ')[-1]}" for j in range(n)],index=i,label_visibility="collapsed")
    j=int(chosen[:2])-1
    if j!=i: st.session_state.slide=j; st.session_state.revealed=False; st.rerun()

pause_class="pause" if st.session_state.paused else ""
extra="" if not st.session_state.revealed else f'<div class="quote"><b>Technical detail</b><br>{html.escape(body)}</div>'

# Main presentation stage.
st.markdown(f'''<div class="stage {pause_class}"><div class="kicker">{html.escape(meta)}</div><h2>{title}</h2><div class="sub">{html.escape(sub)}</div><div class="body">{html.escape(body)}</div><div class="stats"><div class="stat"><div class="v">{html.escape(sv)}</div><div class="l">{html.escape(sl)}</div></div></div><div class="chips">{''.join(f'<span class="chip">{html.escape(x)}</span>' for x in chips)}</div>{extra}</div>''',unsafe_allow_html=True)

# Live model experiment embedded in the presentation.
if i==11:
    st.markdown('<div class="live"><div class="live-title">LIVE MODEL LAB / ACTUAL XGBOOST INFERENCE</div></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        age=st.number_input("Age",18,80,30,key="pres_age")
        income=st.number_input("Annual income",float(df.person_income.min()),float(df.person_income.max()),60000.0,key="pres_income")
        emp=st.number_input("Employment experience",0,60,5,key="pres_emp")
        education=st.selectbox("Education",["High School","Associate","Bachelor","Master","Doctorate"],index=2,key="pres_edu")
        gender=st.selectbox("Gender",["male","female"],key="pres_gender")
        home=st.selectbox("Home ownership",["RENT","OWN","MORTGAGE","OTHER"],key="pres_home")
    with b:
        intent=st.selectbox("Loan intent",["EDUCATION","MEDICAL","VENTURE","PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"],key="pres_intent")
        amount=st.number_input("Loan amount",float(df.loan_amnt.min()),float(df.loan_amnt.max()),10000.0,key="pres_amount")
        rate=st.number_input("Interest rate",float(df.loan_int_rate.min()),float(df.loan_int_rate.max()),10.0,key="pres_rate")
        ratio=st.number_input("Loan / income ratio",float(df.loan_percent_income.min()),min(1.0,float(df.loan_percent_income.max())),0.20,key="pres_ratio")
        history=st.number_input("Credit history length",float(df.cb_person_cred_hist_length.min()),float(df.cb_person_cred_hist_length.max()),4.0,key="pres_history")
        score=st.number_input("Credit score",int(df.credit_score.min()),int(df.credit_score.max()),680,key="pres_score")
        default=st.selectbox("Previous loan default",["No","Yes"],key="pres_default")
    if st.button("RUN REAL MODEL →",type="primary",use_container_width=True):
        applicant={'person_age':age,'person_income':income,'person_home_ownership':home,'person_emp_exp':emp,'loan_intent':intent,'loan_amnt':amount,'loan_int_rate':rate,'loan_percent_income':ratio,'cb_person_cred_hist_length':history,'credit_score':score,'previous_loan_defaults_on_file':default,'person_gender':gender,'person_education':education}
        st.session_state.pres_result=predict_one(bundle,applicant)
    if st.session_state.get("pres_result"):
        r=st.session_state.pres_result; color="#a3e635" if r['prediction']==1 else "#fb7185"
        st.markdown(f'<div class="stage" style="min-height:180px;margin-top:1rem;padding:1.5rem;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.10),transparent 45%),#07101a"><div class="kicker">LIVE OUTPUT</div><h2 style="font-size:2.4rem;color:{color};margin:.3rem 0">{r["label"].upper()}</h2><div class="sub">Approval probability · <b>{r["approval_probability"]:.1%}</b> · Rejection probability · <b>{r["rejection_probability"]:.1%}</b></div></div>',unsafe_allow_html=True)

# Animated benchmark visual on scoreboard slide.
if i==8:
    st.markdown('<div class="bars">'+''.join(f'<div class="barrow"><div class="barname">{name}</div><div class="track"><div class="fill {"win" if name=="XGBoost" else ""}" style="width:{acc}%;animation-delay:{idx*.08}s"></div></div><div class="barval">{acc:.2f}%</div></div>' for idx,(name,acc) in enumerate([("Logistic Regression",86.55),("KNN",86.29),("Decision Tree",89.28),("Random Forest",89.48),("SVM",88.02),("XGBoost",92.87)]))+'</div>',unsafe_allow_html=True)

if i==10:
    cm=metrics['confusion_matrix']
    st.markdown(f'<div class="matrix"><div class="cell"><b>{cm[0][0]:,}</b><small>TRUE NEGATIVE</small></div><div class="cell"><b>{cm[0][1]:,}</b><small>FALSE POSITIVE</small></div><div class="cell"><b>{cm[1][0]:,}</b><small>FALSE NEGATIVE</small></div><div class="cell"><b>{cm[1][1]:,}</b><small>TRUE POSITIVE</small></div></div>',unsafe_allow_html=True)

st.markdown('<div class="small-note">Controls: PREV / NEXT · HOME · PAUSE MOTION · REVEAL DETAIL · JUMP. The Live Model Lab is connected to the reusable inference layer in <code>src/model.py</code>.</div>',unsafe_allow_html=True)
