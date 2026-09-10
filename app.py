from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.model import ENGINEERED_FEATURES, predict_batch, predict_one, train_model

st.set_page_config(page_title="Loan Intelligence", page_icon=None, layout="wide")

st.markdown("""
<style>
.stApp { background:#fff; color:#18324a; }
.block-container { max-width:1220px; padding-top:2rem; padding-bottom:4rem; }
.main-title { font-size:2.55rem; font-weight:750; color:#18324a; letter-spacing:-.03em; }
.sub-title { color:#6b7b8c; font-size:1rem; margin-bottom:1.8rem; }
.section-note { background:#f5f8fb; border:1px solid #e1e8ef; border-radius:12px; padding:.8rem 1rem; color:#526273; }
.value-pill { display:inline-block; background:#eef5ff; border:1px solid #d7e6fb; color:#245a9a; border-radius:8px; padding:3px 9px; font-size:.82rem; font-weight:650; margin-top:-4px; margin-bottom:8px; }
div[data-testid="stMetric"] { background:#f8fafc; border:1px solid #e2e8ef; border-radius:13px; padding:.75rem 1rem; }
div[data-baseweb="select"] > div { background:#fff; border-color:#d5dee7; }
.stButton > button, .stFormSubmitButton > button { background:#2f6f8f; color:#fff; border:0; border-radius:9px; font-weight:650; min-height:2.7rem; }
.stDownloadButton > button { border-radius:9px; border:1px solid #cfd9e2; background:#fff; color:#29465a; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Loan Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered credit decision and model analytics</div>', unsafe_allow_html=True)
DATA_PATH = Path("loan_data.csv")

@st.cache_resource(show_spinner="Preparing HistGradientBoosting for the first prediction...")
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("loan_data.csv was not found in the project folder.")
    return train_model(DATA_PATH)

def slider_with_value(label, min_value, max_value, value, step, formatter, help_text):
    selected = st.slider(label, min_value, max_value, value, step, help=help_text)
    st.markdown(f'<div class="value-pill">Selected: {formatter(selected)}</div>', unsafe_allow_html=True)
    return selected

# Final notebook benchmark: same stratified 80/20 held-out test split.
BENCHMARK = pd.DataFrame([
    ["HistGradientBoosting", .932874, .860537, .8330, .846545, .976527],
    ["XGBoost", .929651, .855434, .8225, .838644, .975957],
    ["Random Forest", .911202, .773079, .8500, .809717, .968021],
    ["Extra Trees", .897755, .738305, .8365, .784341, .963371],
    ["Gradient Boosting", .893532, .717809, .8585, .781876, .962550],
    ["SVM", .877639, .671630, .8795, .761637, .950739],
    ["Decision Tree", .880862, .703152, .8030, .749767, .853058],
    ["Logistic Regression", .861747, .635971, .8840, .739749, .944234],
    ["KNN", .862192, .643396, .8525, .733333, .926571],
], columns=["Model","Accuracy","Precision","Recall","F1 Score","ROC-AUC"]).set_index("Model")

# HistGradientBoosting has no native feature_importances_. These are the
# notebook's permutation-based relative values: 5 repeats on first 1,000 test rows.
TOP_FEATURES = pd.Series({
    "previous_loan_defaults_on_file":31.331987891019153,
    "person_income":11.503531786074674,
    "income_to_loan_ratio":9.485368314833504,
    "person_emp_exp":8.526740665993948,
    "loan_int_rate":8.425832492431889,
    "emp_length_to_age_ratio":6.912209889001011,
    "person_home_ownership":6.861755802219981,
    "loan_intent":3.9354187689202833,
    "cb_person_cred_hist_length":2.623612,
    "credit_score":2.320888,
}, name="Relative Importance (%)")

HGB_CONFUSION = pd.DataFrame([[6728,270],[334,1666]], index=["Actual Rejected (0)","Actual Approved (1)"], columns=["Predicted Rejected (0)","Predicted Approved (1)"])
DATASET_STATS = {"raw_rows":45000,"raw_columns":14,"clean_rows":44988,"features":18,"train":35990,"test":8998,"smote":55980,"class0":35000,"class1":10000,"train0":27990,"train1":8000}

TABS = st.tabs(["Single Applicant", "CSV Analysis", "Model Insights"])

with TABS[0]:
    st.subheader("Applicant profile")
    st.markdown('<div class="section-note">Enter the same 13 raw applicant fields used by the final notebook. The five engineered features are generated automatically by the shared ML core and scored by HistGradientBoosting.</div>', unsafe_allow_html=True)
    with st.form("loan_form"):
        left, right = st.columns(2, gap="large")
        with left:
            age=slider_with_value("Age",18,80,28,1,lambda x:f"{x} years","Applicant age.")
            income=slider_with_value("Annual Income ($)",8000,720000,60000,1000,lambda x:f"${x:,.0f}","Annual income.")
            employment=slider_with_value("Employment Experience (years)",0,50,6,1,lambda x:f"{x} years","Employment experience.")
            loan_amount=slider_with_value("Loan Amount ($)",500,35000,12000,100,lambda x:f"${x:,.0f}","Requested loan amount.")
            interest=slider_with_value("Interest Rate (%)",5.0,20.0,10.0,.1,lambda x:f"{x:.1f}%","Loan interest rate.")
            loan_income=slider_with_value("Loan Percent of Income",0.0,.66,.20,.01,lambda x:f"{x:.0%}","Loan amount as a fraction of annual income.")
        with right:
            credit_history=slider_with_value("Credit History Length (years)",2,30,7,1,lambda x:f"{x} years","Length of credit history.")
            credit_score=slider_with_value("Credit Score",390,850,680,1,lambda x:f"{x}","Credit score.")
            home=st.selectbox("Home Ownership",["RENT","OWN","MORTGAGE","OTHER"])
            education=st.selectbox("Education",["High School","Associate","Bachelor","Master","Doctorate"])
            gender=st.selectbox("Gender",["male","female"])
            defaults=st.selectbox("Previous Loan Defaults",["No","Yes"])
            intent=st.selectbox("Loan Intent",["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
        submitted=st.form_submit_button("Predict Loan Decision", use_container_width=True)
    if submitted:
        try:
            bundle,metrics=load_model()
            applicant={"person_age":int(age),"person_income":int(income),"person_home_ownership":home,"person_emp_exp":int(employment),"loan_intent":intent,"loan_amnt":int(loan_amount),"loan_int_rate":float(interest),"loan_percent_income":float(loan_income),"cb_person_cred_hist_length":int(credit_history),"credit_score":int(credit_score),"previous_loan_defaults_on_file":defaults,"person_gender":gender,"person_education":education}
            result=predict_one(bundle,applicant)
            st.divider(); st.subheader("Prediction result")
            c1,c2,c3=st.columns(3); c1.metric("Decision",result["label"]); c2.metric("Approval Score",f'{result["approval_probability"]*100:.1f}%'); c3.metric("Rejection Score",f'{result["rejection_probability"]*100:.1f}%')
            a,b=st.columns(2)
            with a:
                st.markdown("#### Decision score")
                st.bar_chart(pd.DataFrame({"Score":[result["approval_probability"],result["rejection_probability"]]},index=["Approval","Rejection"]),horizontal=True,height=250)
            with b:
                st.markdown("#### Applicant snapshot")
                st.dataframe(pd.DataFrame({"Field":["Age","Income","Loan Amount","Interest Rate","Loan / Income","Credit Score","Credit History"],"Selected value":[f"{age} years",f"${income:,.0f}",f"${loan_amount:,.0f}",f"{interest:.1f}%",f"{loan_income:.0%}",str(credit_score),f"{credit_history} years"]}),hide_index=True,use_container_width=True)
            st.markdown("#### Runtime HistGradientBoosting metrics")
            q1,q2,q3,q4,q5=st.columns(5)
            q1.metric("Accuracy",f'{metrics["accuracy"]*100:.2f}%'); q2.metric("Precision",f'{metrics["precision"]:.4f}'); q3.metric("Recall",f'{metrics["recall"]:.4f}'); q4.metric("F1",f'{metrics["f1"]:.4f}'); q5.metric("ROC-AUC",f'{metrics["roc_auc"]:.6f}')
            st.markdown("#### Engineered features"); st.dataframe(pd.DataFrame({"Feature":ENGINEERED_FEATURES}),hide_index=True,use_container_width=True)
            st.caption("Probabilities are model scores, not calibrated real-world approval odds.")
        except Exception as exc:
            st.error("The prediction could not be completed."); st.exception(exc)

with TABS[1]:
    st.subheader("CSV analysis dashboard")
    st.markdown('<div class="section-note">Upload applicant records using the same 13-field raw schema. The shared ML core engineers the five derived features and HistGradientBoosting returns predictions and probabilities.</div>', unsafe_allow_html=True)
    sample_path=Path("demo_test.csv")
    if sample_path.exists(): st.download_button("Download Test CSV",data=sample_path.read_bytes(),file_name="loan_prediction_test.csv",mime="text/csv")
    uploaded=st.file_uploader("Upload applicant CSV",type=["csv"])
    if uploaded is not None:
        try:
            df=pd.read_csv(uploaded); m1,m2=st.columns(2); m1.metric("Rows",f"{len(df):,}"); m2.metric("Columns",f"{len(df.columns):,}"); st.dataframe(df.head(10),use_container_width=True)
            if st.button("Analyze CSV",use_container_width=True):
                bundle,_=load_model(); results=predict_batch(bundle,df)
                approved=int((results.prediction==1).sum()); rejected=int((results.prediction==0).sum()); rate=approved/len(results) if len(results) else 0
                st.success(f"Analysis complete: {len(results):,} applicants processed.")
                c1,c2,c3,c4=st.columns(4); c1.metric("Applicants",f"{len(results):,}"); c2.metric("Approved",f"{approved:,}"); c3.metric("Rejected",f"{rejected:,}"); c4.metric("Approval Rate",f"{rate*100:.1f}%")
                x,y=st.columns(2)
                with x: st.markdown("#### Prediction distribution"); st.bar_chart(pd.DataFrame({"Applicants":[approved,rejected]},index=["Approved","Rejected"]),height=300)
                with y: st.markdown("#### Approval score distribution"); st.line_chart(results["approval_probability"].reset_index(drop=True),height=300)
                st.markdown("#### Approval score bands"); bands=pd.cut(results["approval_probability"],bins=[0,.25,.5,.75,1],labels=["0–25%","25–50%","50–75%","75–100%"],include_lowest=True); st.bar_chart(pd.DataFrame({"Applicants":bands.value_counts().sort_index()}),height=280)
                st.markdown("#### Scored rows"); st.dataframe(results,use_container_width=True)
                st.download_button("Download predictions",data=results.to_csv(index=False).encode(),file_name="loan_predictions.csv",mime="text/csv")
        except Exception as exc:
            st.error("The CSV could not be analyzed."); st.exception(exc)

with TABS[2]:
    st.subheader("Model Insights")
    st.markdown('<div class="section-note">Research metrics and diagnostics from <code>final-loan-2.ipynb</code>. HistGradientBoosting is the benchmark leader and the deployable repository model.</div>', unsafe_allow_html=True)
    st.markdown("### Dataset and pipeline")
    cols=st.columns(5)
    for col,label,key in zip(cols,["Raw rows","Clean rows","Train rows","Test rows","Model features"],["raw_rows","clean_rows","train","test","features"]): col.metric(label,f'{DATASET_STATS[key]:,}')
    cols=st.columns(4); cols[0].metric("SMOTE train rows",f'{DATASET_STATS["smote"]:,}'); cols[1].metric("Original class 0",f'{DATASET_STATS["class0"]:,}'); cols[2].metric("Original class 1",f'{DATASET_STATS["class1"]:,}'); cols[3].metric("Class-1 share",f'{DATASET_STATS["class1"]/DATASET_STATS["raw_rows"]*100:.2f}%')
    st.caption("Cleaning keeps age ≤ 80 and employment experience ≤ 50. Five engineered features are added before LabelEncoder, stratified splitting, and SMOTE on training data only.")

    st.markdown("### Nine-model benchmark")
    table=BENCHMARK.copy(); table["Accuracy"]=table.Accuracy.map(lambda x:f"{x*100:.2f}%"); table["Precision"]=table.Precision.map(lambda x:f"{x:.4f}"); table["Recall"]=table.Recall.map(lambda x:f"{x:.4f}"); table["F1 Score"]=table["F1 Score"].map(lambda x:f"{x:.4f}"); table["ROC-AUC"]=table["ROC-AUC"].map(lambda x:f"{x:.6f}"); st.dataframe(table,use_container_width=True)
    a,b=st.columns(2); a.markdown("#### Accuracy leaderboard"); a.bar_chart(BENCHMARK.Accuracy.sort_values()*100,horizontal=True,height=360); b.markdown("#### F1 leaderboard"); b.bar_chart(BENCHMARK["F1 Score"].sort_values()*100,horizontal=True,height=360)

    st.markdown("### HistGradientBoosting diagnostics")
    row=BENCHMARK.loc["HistGradientBoosting"]; m=st.columns(5); m[0].metric("Accuracy",f'{row.Accuracy*100:.2f}%'); m[1].metric("Precision",f'{row.Precision:.4f}'); m[2].metric("Recall",f'{row.Recall:.4f}'); m[3].metric("F1",f'{row["F1 Score"]:.4f}'); m[4].metric("ROC-AUC",f'{row["ROC-AUC"]:.6f}')
    a,b=st.columns(2)
    with a: st.markdown("#### Confusion matrix"); st.dataframe(HGB_CONFUSION,use_container_width=True)
    with b:
        tn,fp,fn,tp=6728,270,334,1666; fpr=fp/(fp+tn); fnr=fn/(fn+tp); err=(fp+fn)/(tn+fp+fn+tp); q=st.columns(4); q[0].metric("FPR",f"{fpr*100:.2f}%"); q[1].metric("FNR",f"{fnr*100:.2f}%"); q[2].metric("Error rate",f"{err*100:.2f}%"); q[3].metric("Correct rate",f"{(1-err)*100:.2f}%")
    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame({"Class":["Rejected (0)","Approved (1)","Macro avg","Weighted avg"],"Precision":[.95270462,.86053719,.90662090,.93221841],"Recall":[.96141755,.833,.89720877,.93287397],"F1 Score":[.95704125,.84654472,.90179298,.93248101],"Support":[6998,2000,8998,8998]}),hide_index=True,use_container_width=True)
    st.markdown("### HistGradientBoosting top features")
    st.caption("Permutation-based relative importance from the notebook's cross-model analysis. HistGradientBoosting has no native feature_importances_.")
    d=TOP_FEATURES.sort_values(); fig,ax=plt.subplots(figsize=(9.5,5.2)); ax.barh(np.arange(len(d)),d.values,height=.62); ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d.index); ax.set_xlabel("Relative Importance (%)"); ax.set_title("Top HistGradientBoosting features",loc="left",fontweight="bold"); ax.grid(axis="x",alpha=.16); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); st.pyplot(fig,clear_figure=True)
    st.dataframe(TOP_FEATURES.rename("Importance (%)").to_frame().style.format("{:.4f}%"),use_container_width=True)
    st.markdown("### HistGradientBoosting configuration")
    st.dataframe(pd.DataFrame({"Parameter":["max_iter","random_state"],"Value":[300,42]}),hide_index=True,use_container_width=True)
    st.markdown("### Separate XGBoost optimization experiment")
    st.caption("Kept because it exists in the source notebook; it is a secondary research experiment, not the deployable model.")
    st.dataframe(pd.DataFrame({"Parameter":["n_estimators","max_depth","learning_rate","subsample","colsample_bytree","gamma"],"Optimal value":[250,12,.01,.60,.80,0]}),hide_index=True,use_container_width=True)
    st.markdown("### Interpretation")
    st.write("HistGradientBoosting is the benchmark leader and the deployable model: 93.2874% accuracy, 0.860537 precision, 0.8330 recall, 0.846545 F1, and 0.976527 ROC-AUC on the held-out test split.")
    st.write("Feature importance here is model attribution rather than causation. The notebook uses permutation importance for HistGradientBoosting because the estimator does not expose native feature_importances_.")
