from pathlib import Path
import html
import random

import pandas as pd
import streamlit as st

from src.model import MODEL_INPUT_COLUMNS, predict_one, train_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "loan_data.csv"

st.set_page_config(page_title="Loan Intelligence — Interactive Deck", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#02050a;--panel:#07101a;--line:#183047;--text:#f7fafc;--muted:#8b9aab;--cyan:#22d3ee;--lime:#a3e635;--purple:#a78bfa;--red:#fb7185;--amber:#fbbf24}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:var(--bg);color:var(--text)}
.block-container{max-width:1580px;padding:1rem 2rem 3rem}#MainMenu,footer{visibility:hidden}
[data-testid="stHeader"]{background:rgba(2,5,10,.92)}
.deck-top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:.3rem 0 .8rem;margin-bottom:1rem}
.deck-brand{font-weight:800;letter-spacing:-.02em}.deck-brand span{color:var(--cyan)}.deck-meta{font-family:'DM Mono';font-size:.58rem;color:#718198;letter-spacing:.1em}
.progress{height:3px;background:#0d1925;border-radius:99px;overflow:hidden;margin-bottom:1.2rem}.progress>div{height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple),var(--lime));transition:width .5s ease}
.kicker{font-family:'DM Mono';color:var(--cyan);font-size:.62rem;letter-spacing:.2em;font-weight:700}.title{font-size:clamp(2.4rem,5.4vw,5.8rem);line-height:.91;letter-spacing:-.075em;font-weight:800;margin:.45rem 0 1rem}.title span{color:var(--cyan)}.sub{max-width:950px;color:#a7b5c5;font-size:.9rem;line-height:1.8}.slide{border:1px solid var(--line);border-radius:30px;background:radial-gradient(circle at 88% 5%,rgba(34,211,238,.10),transparent 26%),radial-gradient(circle at 10% 100%,rgba(167,139,250,.07),transparent 28%),linear-gradient(145deg,#07101a,#040a11);padding:2.4rem;min-height:650px;box-shadow:0 30px 100px rgba(0,0,0,.3);animation:enter .55s ease both;position:relative;overflow:hidden}.slide:after{content:'';position:absolute;right:-170px;bottom:-230px;width:480px;height:480px;border:1px solid rgba(34,211,238,.08);border-radius:50%;box-shadow:0 0 0 40px rgba(34,211,238,.018),0 0 0 90px rgba(34,211,238,.01);pointer-events:none}@keyframes enter{from{opacity:0;transform:translateY(16px) scale(.99)}to{opacity:1;transform:none}}.paused .slide{animation:none}
.stat{font-family:'DM Mono';font-size:3.2rem;font-weight:500;color:var(--lime);letter-spacing:-.08em}.stat-label{color:var(--muted);font-size:.62rem;text-transform:uppercase;letter-spacing:.14em}.card{border:1px solid var(--line);background:rgba(7,16,26,.82);border-radius:18px;padding:1.1rem;height:100%}.card h4{margin:0;font-size:.76rem}.card p{color:var(--muted);font-size:.64rem;line-height:1.65;margin:.45rem 0 0}.mono{font-family:'DM Mono'}.chip{display:inline-block;border:1px solid #21405a;background:#08131e;border-radius:999px;padding:.42rem .62rem;margin:.18rem;color:#b7c5d5;font-family:'DM Mono';font-size:.56rem}.chip b{color:var(--cyan)}.callout{border-left:2px solid var(--cyan);padding:.8rem 1rem;background:rgba(34,211,238,.035);color:#b9c6d4;font-size:.7rem;line-height:1.7;border-radius:0 12px 12px 0}.callout.lime{border-color:var(--lime)}.callout.red{border-color:var(--red)}
.timeline{display:flex;gap:4px;margin:1.2rem 0}.timeline div{height:8px;flex:1;border-radius:99px;background:#122131}.timeline .on{background:linear-gradient(90deg,var(--cyan),var(--lime))}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:.55rem;margin:1.5rem 0}.flow div{padding:1.15rem .8rem;border:1px solid var(--line);border-radius:15px;background:#07111b;text-align:center}.flow b{display:block;font-size:.7rem}.flow small{display:block;color:var(--muted);font-size:.55rem;margin-top:.35rem}.arrow{color:var(--cyan);font-size:1.2rem;display:flex;align-items:center;justify-content:center}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;margin:1rem 0}.metric{border:1px solid var(--line);border-radius:15px;background:#08111b;padding:.9rem}.metric .n{font-family:'DM Mono';font-size:1.35rem}.metric .l{font-size:.53rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-top:.25rem}.bar{margin:.5rem 0}.barline{height:10px;background:#122131;border-radius:99px;overflow:hidden}.barline i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--lime));border-radius:99px}.barlabel{display:flex;justify-content:space-between;font-size:.58rem;color:#aebaca;margin-bottom:.22rem}.xgb{border:1px solid rgba(163,230,53,.35);background:rgba(163,230,53,.035);border-radius:20px;padding:1.2rem}.big-decision{text-align:center;border:1px solid var(--line);border-radius:22px;padding:2rem;background:radial-gradient(circle at 50% 0%,rgba(34,211,238,.12),transparent 50%),#06101a}.big-decision .label{font-size:3.1rem;line-height:1;font-weight:800;letter-spacing:-.06em}.approved{color:var(--lime)}.rejected{color:var(--red)}.prob{font-family:'DM Mono';font-size:1.2rem;margin-top:.7rem}.note{font-size:.58rem;color:#65768b;line-height:1.6}.kbd{font-family:'DM Mono';font-size:.55rem;border:1px solid #29435b;padding:.2rem .35rem;border-radius:5px;color:#b7c8d8;background:#08121c}
.stButton>button{border:1px solid #24445c;background:linear-gradient(135deg,#091823,#07111a);color:#e6fbff;border-radius:11px;font-weight:800;min-height:42px}.stButton>button:hover{border-color:var(--cyan);transform:translateY(-1px);box-shadow:0 10px 30px rgba(34,211,238,.1)}
[data-testid="stSidebar"]{background:#040911;border-right:1px solid var(--line)}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def boot():
    return train_model(DATA)

@st.cache_data(show_spinner=False)
def data():
    return pd.read_csv(DATA)

bundle, metrics = boot()
df = data()

if "deck_slide" not in st.session_state: st.session_state.deck_slide = 1
if "deck_paused" not in st.session_state: st.session_state.deck_paused = False
if "demo_result" not in st.session_state: st.session_state.demo_result = None

slides = [
("01 / OPENING","Loan <span>Intelligence.</span>","The project, as a product.","Start with the end state: a complete ML workflow that can be explored, challenged, and demonstrated live.","92.87%","held-out test accuracy"),
("02 / THE STORY","From raw rows to a <span>decision engine.</span>","A 26-chapter technical narrative.","We move through problem framing, EDA, feature engineering, imbalance handling, model selection, diagnostics, deployment, and live experimentation.",None,None),
("03 / PROBLEM","What are we actually <span>predicting?</span>","Binary classification: loan status.","Given applicant and loan characteristics, predict the target class. The objective is not to claim a perfect underwriting system; it is to demonstrate a traceable supervised-learning workflow.",None,None),
("04 / DATA","Meet the <span>data.</span>","45K+ source rows · 13 raw model inputs.",f"The dataset combines demographic, employment, loan, credit, ownership, intent, and default-history signals. After the project cleaning workflow, 44,990 rows remain for modeling.",f"{len(df):,}","source rows"),
("05 / DATA EXPLORATION","Before modeling: <span>look.</span>","Structure → distributions → relationships.","EDA checks shape, columns, information types, duplicates, missingness, numeric distributions, categorical counts, correlations, and target-related behavior.",None,None),
("06 / QUALITY","Make the signal <span>stable.</span>","IQR clipping + domain-style filters used by the project.","Selected numeric variables are clipped using IQR bounds. The workflow also filters person_age ≤ 80 and person_emp_exp ≤ 60 before encoding.",None,None),
("07 / REPRESENTATION","Raw categories become <span>features.</span>","Binary maps + ordinal education + one-hot categories.","Gender and previous-default history are binary mapped. Education is ordinally encoded. Home ownership and loan intent become one-hot columns.",None,None),
("08 / SCALING","Why <span>RobustScaler?</span>","Scale without letting large magnitudes dominate.","The scaler is fitted on the training split and then applied to the held-out test data. This keeps the inference representation aligned with the trained model.",None,None),
("09 / IMBALANCE","The target was <span>skewed.</span>","27,991 class-0 vs 8,001 class-1 examples before the split-level training workflow.","SMOTETomek is applied to the training data after scaling. The test set is kept untouched for evaluation.","27,887 × 2","balanced training classes"),
("10 / MODEL ARENA","Six models. <span>One benchmark.</span>","Same held-out test split. Multiple metrics.","Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and XGBoost compete on accuracy plus class-1 precision, recall, and F1.",None,None),
("11 / THE WINNER","XGBoost reaches <span>92.87%.</span>","Best overall benchmark result in this experiment.","XGBoost leads test accuracy, class-1 precision, and class-1 F1. Logistic Regression and SVM still have higher class-1 recall, so the selection is a trade-off—not a universal truth.","92.87%","XGBoost test accuracy"),
("12 / LIVE LAB","Stop the deck. <span>Run the model.</span>","Interactive experiment — real inference.","Change the applicant below and run the exact reusable XGBoost inference path. This slide is deliberately a working laboratory, not a fake animation.",None,None),
("13 / DECISION","A probability is not a <span>verdict.</span>","Interpret model output with context.","The UI exposes predicted class and model probability. In a real lending setting, threshold choice, calibration, policy rules, fairness, and human oversight would matter.",None,None),
("14 / CONFUSION MATRIX","Where did the model <span>miss?</span>","Errors are more informative than accuracy alone.","XGBoost test confusion matrix: 6762 true class-0, 237 false class-1, 405 false class-0, and 1594 true class-1 predictions.",None,None),
("15 / PRECISION vs RECALL","There is a <span>trade-off.</span>","Different models optimize different failure patterns.","XGBoost has class-1 precision 0.87, recall 0.80, F1 0.83. Logistic Regression and SVM reach 0.92 recall in this benchmark.",None,None),
("16 / OVERFITTING","Training score can <span>lie.</span>","Compare train behavior with held-out behavior.","Decision Tree reaches 100% training accuracy but 89.28% test accuracy. XGBoost reaches 97.68% training accuracy and 92.87% test accuracy.",None,None),
("17 / WHAT-IF","What changes the <span>prediction?</span>","A controlled feature perturbation experiment.","Start from one applicant, change one or two variables, rerun the real model, and compare the probability. This demonstrates model sensitivity without claiming causality.",None,None),
("18 / FEATURE IMPORTANCE","Open the <span>black box.</span>","Model-level feature importance — not causal importance.","Inspect the features XGBoost relied on most in this fitted run. The visualization is sorted and scaled for readability.",None,None),
("19 / DATA EXPLORER","Bring the dataset <span>on stage.</span>","Random rows, target distribution, and quick inspection.","Use the live table to select a random slice from the actual CSV and inspect the inputs the model sees. No screenshot is being used here—the app reads the repository data.",None,None),
("20 / PRODUCT ARCHITECTURE","Notebook → <span>product.</span>","Research, reusable ML core, UI, tests, CI.","The notebook documents the experiment. src/model.py owns reusable training and inference. Streamlit becomes the product layer. Tests and GitHub Actions protect the contract.",None,None),
("21 / BATCH LAB","One applicant is cool. <span>1000 is better.</span>","Batch inference turns the demo into a workflow.","Upload a CSV using the project's schema, validate it, run batch predictions, inspect probabilities, optionally compare labelled rows, and download scored output.",None,None),
("22 / EXPLAINABILITY","Make the result <span>defensible.</span>","What we can explain today—and what we cannot.","Feature importance can describe model reliance. It does not prove causality, fairness, or that a feature should be used in production. Those require separate validation.",None,None),
("23 / RESPONSIBLE ML","A loan model affects <span>people.</span>","The demo is educational, not underwriting policy.","No fairness audit, calibration study, drift monitoring, or production validation has been completed. The project explicitly documents these boundaries.",None,None),
("24 / PRODUCTION","From demo to <span>system.</span>","Versioning → API → monitoring → governance.","Next steps include versioned artifacts, FastAPI, SHAP, calibration, subgroup evaluation, Docker, MLflow, model registry, drift monitoring, observability, and CI/CD.",None,None),
("25 / THE TAKEAWAY","The score is only <span>one part.</span>","The real deliverable is the system around it.","A reproducible pipeline, honest benchmark, reusable inference, interactive experiments, batch workflow, documentation, and a clear production roadmap make the project defensible.","RESEARCH → PRODUCT","the full ML journey"),
("26 / CLOSE","Built to be <span>challenged.</span>","Yusuf Ayman Tolba · Mohamed Reda Hussein · Abdelrahman Mohamed Ahmed · Abdelmoniem Ibrahim Abdelmoniem","NTI Machine Learning Track — Loan Intelligence. Open the app, change the data, and see what the model does.","LIVE","DEMO READY"),
]

n = st.session_state.deck_slide
paused = st.session_state.deck_paused
st.markdown(f'<div class="deck-top"><div class="deck-brand">◈ LOAN <span>INTELLIGENCE</span></div><div class="deck-meta">NTI / MACHINE LEARNING / INTERACTIVE PRESENTATION</div></div>', unsafe_allow_html=True)
st.markdown(f'<div class="progress"><div style="width:{n/len(slides)*100:.2f}%"></div></div>', unsafe_allow_html=True)

# controls
c1,c2,c3,c4,c5,c6 = st.columns([1,1,1.5,2,1.3,1.2])
with c1:
    if st.button("← PREV", use_container_width=True): st.session_state.deck_slide=max(1,n-1); st.rerun()
with c2:
    if st.button("NEXT →", use_container_width=True): st.session_state.deck_slide=min(len(slides),n+1); st.rerun()
with c3:
    target=st.selectbox("Jump", list(range(1,len(slides)+1)), index=n-1, label_visibility="collapsed")
    if target!=n: st.session_state.deck_slide=target; st.rerun()
with c4:
    if st.button("▶ / ⏸  PAUSE MOTION" if not paused else "▶ RESUME MOTION", use_container_width=True): st.session_state.deck_paused=not paused; st.rerun()
with c5:
    st.caption(f"SLIDE {n:02d} / {len(slides):02d}")
with c6:
    st.caption("⌨ arrows / buttons")

k,title,sub,body,stat,stat_label = slides[n-1]
cls = "slide paused" if paused else "slide"
st.markdown(f'<div class="{cls}"><div class="kicker">{k}</div><div class="title">{title}</div><div class="sub">{sub}</div><div style="margin-top:1.25rem;max-width:1050px;color:#b4c1cf;font-size:.78rem;line-height:1.8">{body}</div></div>', unsafe_allow_html=True)

# rich slide content
if n == 1:
    a,b,c=st.columns([1.2,1,1])
    with a: st.markdown(f'<div class="stat">92.87%</div><div class="stat-label">held-out test accuracy</div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="card"><h4>REAL MODEL</h4><p>XGBoost inference is wired to the reusable ML core.</p></div>',unsafe_allow_html=True)
    with c: st.markdown('<div class="card"><h4>LIVE EXPERIENCE</h4><p>Experiment with applicants and CSV batches without leaving the app.</p></div>',unsafe_allow_html=True)
    st.image(str(ROOT/'assets'/'loan-intelligence-flow.svg'), use_container_width=True)
elif n == 2:
    st.markdown('<div class="timeline">'+''.join(f'<div class="on"></div>' if i<5 else '<div></div>' for i in range(9))+'</div>',unsafe_allow_html=True)
    cols=st.columns(3)
    items=[("RESEARCH","EDA + preprocessing + benchmark"),("ENGINEERING","reusable model + batch inference"),("PRODUCT","interactive UI + presentation")]
    for col,(h,p) in zip(cols,items):
        with col: st.markdown(f'<div class="card"><h4>{h}</h4><p>{p}</p></div>',unsafe_allow_html=True)
elif n == 3:
    cols=st.columns(4)
    for col,(a,b) in zip(cols,[("INPUT","Applicant + loan features"),("TARGET","loan_status"),("TYPE","Binary classification"),("OUTPUT","Class + probability")]):
        with col: st.markdown(f'<div class="card"><h4>{a}</h4><p>{b}</p></div>',unsafe_allow_html=True)
elif n == 4:
    cols=st.columns(4)
    for col,(a,b) in zip(cols,[("45K+","source rows"),("44,990","rows after project cleaning"),("13","raw model inputs"),("20","encoded model features")]):
        with col: st.markdown(f'<div class="metric"><div class="n">{a}</div><div class="l">{b}</div></div>',unsafe_allow_html=True)
elif n == 5:
    sample=df.sample(min(8,len(df)),random_state=42)
    st.dataframe(sample, use_container_width=True, height=330, hide_index=True)
elif n == 6:
    st.markdown('<div class="flow"><div><b>RAW</b><small>outliers / ranges</small></div><div><b>IQR</b><small>clip selected numeric</small></div><div><b>FILTER</b><small>age ≤ 80 / exp ≤ 60</small></div><div><b>ENCODE</b><small>categorical → numeric</small></div><div><b>READY</b><small>model features</small></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="callout">Important: this presentation describes the project workflow as implemented. Production hardening would fit preprocessing statistics strictly inside the training split.</div>',unsafe_allow_html=True)
elif n == 7:
    cols=st.columns(4)
    for col,(a,b) in zip(cols,[("GENDER","male/female → 1/0"),("DEFAULT","Yes/No → 1/0"),("EDUCATION","High School → Doctorate"),("CATEGORICAL","one-hot ownership + intent")]):
        with col: st.markdown(f'<div class="card"><h4>{a}</h4><p>{b}</p></div>',unsafe_allow_html=True)
elif n == 8:
    st.markdown('<div class="flow"><div><b>TRAIN SPLIT</b><small>fit scaler</small></div><div><b>ROBUSTSCALER</b><small>transform train</small></div><div><b>SMOTETOMEK</b><small>balance train</small></div><div><b>TEST</b><small>transform only</small></div><div><b>XGBOOST</b><small>learn patterns</small></div></div>',unsafe_allow_html=True)
elif n == 9:
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="bar"><div class="barlabel"><span>CLASS 0</span><b>27,991</b></div><div class="barline"><i style="width:100%"></i></div></div><div class="bar"><div class="barlabel"><span>CLASS 1</span><b>8,001</b></div><div class="barline"><i style="width:28.6%"></i></div></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="xgb"><div class="stat">27,887 × 2</div><div class="stat-label">after SMOTETomek</div><p class="note">Applied to training data only. The held-out test distribution is not resampled.</p></div>',unsafe_allow_html=True)
elif n == 10:
    st.image(str(ROOT/'assets'/'model-arena.svg'), use_container_width=True)
    st.dataframe(pd.DataFrame([["Logistic Regression",86.55,.64,.92,.75],["KNN",86.29,.64,.88,.74],["Decision Tree",89.28,.73,.82,.77],["Random Forest",89.48,.71,.88,.79],["SVM",88.02,.67,.92,.77],["XGBoost",92.87,.87,.80,.83]],columns=["Model","Accuracy","Class-1 Precision","Class-1 Recall","Class-1 F1"]),use_container_width=True,hide_index=True)
elif n == 11:
    a,b=st.columns([1,2])
    with a: st.markdown('<div class="xgb"><div class="stat">92.87%</div><div class="stat-label">test accuracy</div><div class="metric-grid"><div class="metric"><div class="n">0.87</div><div class="l">precision</div></div><div class="metric"><div class="n">0.80</div><div class="l">recall</div></div><div class="metric"><div class="n">0.83</div><div class="l">F1</div></div></div></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="callout lime"><b>Why selected?</b><br>XGBoost leads the benchmark on test accuracy, class-1 precision, and class-1 F1. It does not lead recall; that trade-off stays visible.</div>',unsafe_allow_html=True)
elif n == 12:
    st.markdown('<div class="callout lime">LIVE MODEL LAB · Change inputs, press RUN, and the result comes from <b>predict_one()</b>.</div>',unsafe_allow_html=True)
    left,right=st.columns([1.35,.8])
    with left:
        age=st.slider("Age",18,80,30,key="lab_age")
        income=st.number_input("Annual income",float(df.person_income.min()),float(df.person_income.max()),float(df.person_income.median()),step=500.0,key="lab_income")
        emp=st.slider("Employment experience",0,60,5,key="lab_emp")
        amount=st.number_input("Loan amount",float(df.loan_amnt.min()),float(df.loan_amnt.max()),float(df.loan_amnt.median()),step=500.0,key="lab_amount")
        rate=st.number_input("Interest rate",float(df.loan_int_rate.min()),float(df.loan_int_rate.max()),float(df.loan_int_rate.median()),step=.1,key="lab_rate")
        lpi=st.number_input("Loan percent income",float(df.loan_percent_income.min()),1.0,float(min(.2,df.loan_percent_income.median())),step=.01,key="lab_lpi")
    with right:
        gender=st.selectbox("Gender",["male","female"],key="lab_gender")
        edu=st.selectbox("Education",["High School","Associate","Bachelor","Master","Doctorate"],index=2,key="lab_edu")
        home=st.selectbox("Home ownership",["RENT","OWN","MORTGAGE","OTHER"],key="lab_home")
        intent=st.selectbox("Loan intent",["EDUCATION","MEDICAL","VENTURE","PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"],key="lab_intent")
        hist=st.number_input("Credit history length",float(df.cb_person_cred_hist_length.min()),float(df.cb_person_cred_hist_length.max()),float(df.cb_person_cred_hist_length.median()),step=.1,key="lab_hist")
        score=st.number_input("Credit score",int(df.credit_score.min()),int(df.credit_score.max()),int(df.credit_score.median()),key="lab_score")
        default=st.selectbox("Previous default",["No","Yes"],key="lab_default")
        if st.button("⚡ RUN REAL MODEL",use_container_width=True):
            st.session_state.demo_result=predict_one(bundle,dict(person_age=age,person_income=income,person_home_ownership=home,person_emp_exp=emp,loan_intent=intent,loan_amnt=amount,loan_int_rate=rate,loan_percent_income=lpi,cb_person_cred_hist_length=hist,credit_score=score,previous_loan_defaults_on_file=default,person_gender=gender,person_education=edu))
    r=st.session_state.demo_result
    if r:
        cls='approved' if r['prediction']==1 else 'rejected'
        st.markdown(f'<div class="big-decision"><div class="label {cls}">{html.escape(r["label"]).upper()}</div><div class="prob">{r["approval_probability"]:.1%} approval probability</div><div class="note">rejection probability: {r["rejection_probability"]:.1%}</div></div>',unsafe_allow_html=True)
elif n == 13:
    st.markdown('<div class="metric-grid"><div class="metric"><div class="n">CLASS</div><div class="l">discrete prediction</div></div><div class="metric"><div class="n">P(y=1)</div><div class="l">approval probability</div></div><div class="metric"><div class="n">P(y=0)</div><div class="l">rejection probability</div></div><div class="metric"><div class="n">POLICY</div><div class="l">not included</div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="callout">The probability is a model output. It should not be presented as a calibrated real-world probability of loan approval without calibration and external validation.</div>',unsafe_allow_html=True)
elif n == 14:
    cm=metrics["confusion_matrix"]
    st.markdown(f'''<div class="metric-grid"><div class="metric"><div class="n">{cm[0][0]:,}</div><div class="l">true class 0</div></div><div class="metric"><div class="n">{cm[0][1]:,}</div><div class="l">false class 1</div></div><div class="metric"><div class="n">{cm[1][0]:,}</div><div class="l">false class 0</div></div><div class="metric"><div class="n">{cm[1][1]:,}</div><div class="l">true class 1</div></div></div>''',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(cm,index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]),use_container_width=True)
elif n == 15:
    chart=pd.DataFrame({"Model":["Logistic Regression","KNN","Decision Tree","Random Forest","SVM","XGBoost"],"Precision":[.64,.64,.73,.71,.67,.87],"Recall":[.92,.88,.82,.88,.92,.80]})
    st.bar_chart(chart.set_index("Model"),use_container_width=True,height=360)
    st.markdown('<div class="callout">If missing a true class-1 case is more costly, recall matters. If false class-1 predictions are more costly, precision matters. The project does not choose a universal threshold policy.</div>',unsafe_allow_html=True)
elif n == 16:
    st.markdown('<div class="metric-grid"><div class="metric"><div class="n">100%</div><div class="l">Decision Tree train</div></div><div class="metric"><div class="n">89.28%</div><div class="l">Decision Tree test</div></div><div class="metric"><div class="n">97.68%</div><div class="l">XGBoost train</div></div><div class="metric"><div class="n">92.87%</div><div class="l">XGBoost test</div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="callout red">A high training score is not the goal. The held-out test result is the evidence used for model comparison.</div>',unsafe_allow_html=True)
elif n == 17:
    base={"person_age":30,"person_income":50000.0,"person_home_ownership":"RENT","person_emp_exp":5,"loan_intent":"EDUCATION","loan_amnt":10000.0,"loan_int_rate":10.0,"loan_percent_income":.20,"cb_person_cred_hist_length":5.0,"credit_score":680,"previous_loan_defaults_on_file":"No","person_gender":"male","person_education":"Bachelor"}
    a,b=st.columns(2)
    with a:
        change=st.slider("Interest rate",float(df.loan_int_rate.min()),float(df.loan_int_rate.max()),10.0,.1,key="whatif_rate")
        income2=st.number_input("Income",float(df.person_income.min()),float(df.person_income.max()),50000.0,500.0,key="whatif_income")
        if st.button("RUN WHAT-IF",use_container_width=True):
            one=base.copy(); one["loan_int_rate"]=change; one["person_income"]=income2
            st.session_state.whatif=predict_one(bundle,one)
    with b:
        r=st.session_state.get("whatif")
        if r: st.markdown(f'<div class="big-decision"><div class="label {"approved" if r["prediction"] else "rejected"}">{r["label"].upper()}</div><div class="prob">{r["approval_probability"]:.1%}</div></div>',unsafe_allow_html=True)
        else: st.markdown('<div class="card"><h4>CONTROLLED EXPERIMENT</h4><p>Change variables and rerun. Compare the probability. This is model sensitivity, not a causal experiment.</p></div>',unsafe_allow_html=True)
elif n == 18:
    imp=pd.DataFrame({"feature":bundle.feature_columns,"importance":bundle.model.feature_importances_}).sort_values("importance",ascending=False).head(12).set_index("feature")
    st.bar_chart(imp, use_container_width=True, height=400)
    st.markdown('<div class="callout">Important: feature importance describes model reliance in this fitted XGBoost model. It is <b>not</b> a causal ranking and should not be interpreted as “this feature causes approval/rejection.”</div>',unsafe_allow_html=True)
elif n == 19:
    seed=st.number_input("Random seed",0,9999,42,key="explorer_seed")
    rows=st.slider("Rows",5,30,10,key="explorer_rows")
    st.dataframe(df.sample(rows,random_state=seed),use_container_width=True,height=420,hide_index=True)
elif n == 20:
    st.markdown('<div class="flow"><div><b>loan.ipynb</b><small>research / EDA</small></div><div><b>src/model.py</b><small>ML core</small></div><div><b>app.py</b><small>product UI</small></div><div><b>tests/</b><small>quality contract</small></div><div><b>CI</b><small>automated checks</small></div></div>',unsafe_allow_html=True)
    st.image(str(ROOT/'assets'/'loan-intelligence-flow.svg'),use_container_width=True)
elif n == 21:
    st.markdown('<div class="callout lime"><b>Batch Lab lives in the main app.</b> Use it to upload the raw schema, validate, score rows, and download the result. This presentation slide explains the workflow so you can demo it live next.</div>',unsafe_allow_html=True)
    st.code("CSV → schema validation → predict_batch() → probabilities → optional evaluation → scored CSV",language="text")
    st.markdown('<div class="metric-grid"><div class="metric"><div class="n">1</div><div class="l">upload</div></div><div class="metric"><div class="n">N</div><div class="l">rows scored</div></div><div class="metric"><div class="n">2</div><div class="l">probabilities</div></div><div class="metric"><div class="n">CSV</div><div class="l">download</div></div></div>',unsafe_allow_html=True)
elif n == 22:
    cols=st.columns(3)
    for col,(h,p) in zip(cols,[("WHAT WE HAVE","feature importance + probabilities + benchmark"),("WHAT WE DO NOT HAVE","causal explanations or fairness proof"),("NEXT","SHAP + calibration + subgroup analysis")]):
        with col: st.markdown(f'<div class="card"><h4>{h}</h4><p>{p}</p></div>',unsafe_allow_html=True)
elif n == 23:
    st.markdown('<div class="callout red"><b>Responsible-use boundary:</b> this project is an educational / portfolio ML demonstration, not a production credit-underwriting system.</div>',unsafe_allow_html=True)
    st.markdown('<div class="metric-grid"><div class="metric"><div class="n">NO</div><div class="l">fairness audit</div></div><div class="metric"><div class="n">NO</div><div class="l">calibration study</div></div><div class="metric"><div class="n">NO</div><div class="l">drift monitoring</div></div><div class="metric"><div class="n">YES</div><div class="l">documented limits</div></div></div>',unsafe_allow_html=True)
elif n == 24:
    st.markdown('<div class="flow"><div><b>ARTIFACT</b><small>version model</small></div><div><b>API</b><small>FastAPI</small></div><div><b>EXPLAIN</b><small>SHAP</small></div><div><b>MONITOR</b><small>drift / metrics</small></div><div><b>GOVERN</b><small>registry / CI/CD</small></div></div>',unsafe_allow_html=True)
elif n == 25:
    st.markdown('<div class="xgb"><div class="stat">RESEARCH → PRODUCT</div><div class="stat-label">the real deliverable</div><p class="note">Data quality, preprocessing, imbalance handling, benchmark discipline, reusable inference, interactive experimentation, batch scoring, documentation, testing, and a production roadmap.</p></div>',unsafe_allow_html=True)
elif n == 26:
    st.markdown('<div class="callout lime"><b>LIVE DEMO SCRIPT:</b> open Prediction Studio → enter an applicant → run model → return here → open Model Insights → show benchmark → use Batch Lab.</div>',unsafe_allow_html=True)
    st.markdown('<div style="margin-top:1.5rem;font-size:.75rem;color:#9cacbd;line-height:1.8"><b>Team</b><br>Yusuf Ayman Tolba · Mohamed Reda Hussein · Abdelrahman Mohamed Ahmed · Abdelmoniem Ibrahim Abdelmoniem</div>',unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#4e6175;font:500 .55rem DM Mono;margin-top:1.4rem;letter-spacing:.1em">LOAN INTELLIGENCE · NTI MACHINE LEARNING · INTERACTIVE DECK</div>',unsafe_allow_html=True)
