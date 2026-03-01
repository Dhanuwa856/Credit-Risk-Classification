import streamlit as st
import pickle
import pandas as pd

# 1. App එකේ Title එක
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Credit Risk Prediction App")
st.write(
    "මෙම පද්ධතිය හරහා පාරිභෝගිකයාගේ තොරතුරු ඇතුළත් කර, ඔහු/ඇය ණය ගෙවීම පැහැර හරීද (Defaulter) යන්න 96% ක නිරවද්‍යතාවයකින් අනුමාන කළ හැක.")


# 2. Model Load කරගැනීම
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

pipeline = load_model()

# 💡 The Pro Trick: වෙනම pkl ෆයිල් එකක් නැතුව, Pipeline එකෙන්ම Columns ටික ඇදලා ගමු!
model_columns = list(pipeline.feature_names_in_)

st.success("AI Model Loaded Successfully! 🚀")
st.markdown("---")

# 3. පාරිභෝගික තොරතුරු ලබා ගැනීම (UI)
st.subheader("📝 පාරිභෝගික තොරතුරු ඇතුළත් කරන්න")
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("වයස (Age)", min_value=18, max_value=100, value=30)
    person_income = st.number_input("වාර්ෂික ආදායම ($)", min_value=0, value=50000)
    person_emp_length = st.number_input("රැකියාවේ නියුතු කාලය (අවුරුදු)", min_value=0.0, value=5.0)
    loan_amnt = st.number_input("ණය මුදල ($)", min_value=0, value=10000)

with col2:
    loan_int_rate = st.number_input("පොලී අනුපාතය (%)", min_value=0.0, value=10.0)
    person_home_ownership = st.selectbox("නිවාස අයිතිය", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox("ණය ලබාගන්නා හේතුව",
                               ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    cb_person_default_on_file = st.selectbox("පෙර ණය පැහැර හැර තිබේද?", ["Y", "N"])

st.markdown("---")

# 4. Prediction එක සිදු කිරීම
if st.button("🔍 ණය අවදානම පරීක්ෂා කරන්න (Predict)"):

    # පාරිභෝගිකයාගේ දත්ත Dictionary එකකට ගැනීම
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'cb_person_default_on_file': cb_person_default_on_file,
        # අනෙකුත් අවශ්‍ය දත්ත සඳහා සාමාන්‍ය අගයන් ලබා දීම
        'loan_grade': 'A',
        'loan_percent_income': loan_amnt / person_income if person_income > 0 else 0,
        'cb_person_cred_hist_length': 2
    }

    # Dataframe එකක් බවට පත් කර, Dummy Variables සෑදීම
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)

    # 💡 The Magic Trick: පුහුණු කළ Model එකේ Columns වලටම අලුත් දත්ත ටික ගැලපීම (අඩුවන ඒවාට 0 දැමීම)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # අනුමානය ලබා ගැනීම
    prediction = pipeline.predict(input_encoded)

    # ප්‍රතිඵලය ප්‍රදර්ශනය කිරීම
    if prediction[0] == 1:
        st.error("⚠️ අවදානමක් ඇත! (High Risk of Default) - මෙම පාරිභෝගිකයා ණය පැහැර හැරීමට ඉඩ ඇත.")
    else:
        st.success("✅ අවදානමක් නැත! (Low Risk) - ණය මුදල අනුමත කළ හැක.")
        st.balloons()  # පොඩි Celebration එකක්! 🎉