import streamlit as st
import pickle
import pandas as pd

# 1. App Title and Description
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Credit Risk Prediction App")
st.write(
    "Enter the customer's details below to predict the likelihood of a loan default with 96% precision."
)

# 2. Load Model and Feature Columns
@st.cache_resource
def load_model():
    # Loading the pre-trained machine learning pipeline
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

pipeline = load_model()

# Extract expected columns directly from the trained pipeline
model_columns = list(pipeline.feature_names_in_)

st.success("AI Model Loaded Successfully! 🚀")
st.markdown("---")

# 3. Customer Information Input (UI)
st.subheader("📝 Enter Customer Information")
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, value=5.0)
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)

with col2:
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox("Loan Intent",
                               ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    cb_person_default_on_file = st.selectbox("Prior Default on File?", ["Y", "N"])

st.markdown("---")

# 4. Prediction Logic
if st.button("🔍 Predict Credit Risk"):

    # Store customer data in a dictionary for processing
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'cb_person_default_on_file': cb_person_default_on_file,
        'loan_grade': 'A',
        'loan_percent_income': loan_amnt / person_income if person_income > 0 else 0,
        'cb_person_cred_hist_length': 2
    }

    # Convert to DataFrame and handle dummy variables
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)

    # Reindex to match the model's expected feature set
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Generate Prediction using the Random Forest model
    prediction = pipeline.predict(input_encoded)

    # Display Results
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Default! - This customer is likely to default on the loan.")
    else:
        st.success("✅ Low Risk! - The loan can be safely approved.")
        st.balloons()

# --- 5. Professional Footer Section ---
st.markdown("<br><br>", unsafe_allow_html=True) # මෙතන unsafe_allow_html ලෙස වෙනස් විය යුතුයි
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Developed with ❤️ by <b>Dhanushka Rathnayaka</b></p>
        <p><a href="https://www.dhanushka.live/" target="_blank">🌐 Visit My Portfolio Website</a></p>
    </div>
    """,
    unsafe_allow_html=True # මෙතනත් unsafe_allow_html ලෙස වෙනස් විය යුතුයි
)