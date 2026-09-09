from pathlib import Path

import streamlit as st

from src.model import predict_one, train_model


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_data.csv"

st.set_page_config(
    page_title="Loan Intelligence",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #070b14; color: #f4f7fa; }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    .hero {
        padding: 2rem 2.2rem;
        border: 1px solid #20304a;
        border-radius: 22px;
        background: linear-gradient(135deg, #0c1424 0%, #0b1020 60%, #12102a 100%);
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 3rem; letter-spacing: -0.04em; }
    .hero p { color: #9aa8bb; font-size: 1.05rem; margin-top: .6rem; }
    .result {
        padding: 1.6rem;
        border-radius: 18px;
        border: 1px solid #20304a;
        background: #0c1424;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Loan Intelligence</h1>
      <p>NTI Machine Learning • XGBoost loan-status prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the benchmark model…")
def get_model():
    return train_model(DATA_PATH)


bundle, metrics = get_model()

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Applicant profile")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 80, 30)
        income = st.number_input("Annual income", 0.0, 2_000_000.0, 60000.0)
        emp_exp = st.number_input("Employment experience", 0, 60, 5)
        education = st.selectbox(
            "Education",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
            index=2,
        )
        gender = st.selectbox("Gender", ["male", "female"])

    with c2:
        home = st.selectbox(
            "Home ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
        )
        intent = st.selectbox(
            "Loan intent",
            [
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "DEBTCONSOLIDATION",
                "HOMEIMPROVEMENT",
            ],
        )
        loan_amount = st.number_input("Loan amount", 0.0, 500000.0, 10000.0)
        interest = st.number_input("Interest rate (%)", 0.0, 50.0, 10.0)
        loan_percent_income = st.number_input(
            "Loan percent of income", 0.0, 1.0, 0.20, step=0.01
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        credit_history = st.number_input(
            "Credit history length", 0.0, 50.0, 5.0
        )
    with c4:
        credit_score = st.number_input("Credit score", 300, 850, 680)
    with c5:
        previous_default = st.selectbox(
            "Previous loan default", ["No", "Yes"]
        )

    if st.button("Run prediction", type="primary", use_container_width=True):
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
        st.session_state["result"] = predict_one(bundle, applicant)

with right:
    st.subheader("Decision")

    if "result" not in st.session_state:
        st.markdown(
            '<div class="result"><h2>Ready</h2>'
            '<p>Enter applicant information and run the model.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        result = st.session_state["result"]
        probability = result["approval_probability"]
        st.markdown(
            f'''<div class="result">
              <h1>{result["label"]}</h1>
              <h2>{probability:.1%} approval probability</h2>
              <p>Model: XGBoost • benchmark test accuracy: {metrics["accuracy"]:.2%}</p>
            </div>''',
            unsafe_allow_html=True,
        )

    st.divider()
    a, b = st.columns(2)
    with a:
        st.metric("Test accuracy", f'{metrics["accuracy"]:.2%}')
    with b:
        st.metric("Test rows", f'{metrics["test_rows"]:,}')

st.caption(
    "Educational demonstration only. The prediction is not financial advice "
    "or an automated lending decision."
)
