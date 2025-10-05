import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="📈",
    layout="wide"
)

# --- MODEL AND SCALER LOADING ---
@st.cache_resource
def load_artifacts():
    """Loads the pickled model and scaler."""
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Make sure 'model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_artifacts()

# --- DATA MAPPINGS (FROM NOTEBOOK ANALYSIS) ---
# These dictionaries will convert user-friendly inputs to the numeric codes the model expects.
gender_map = {'F': 0, 'M': 1}
yes_no_map = {'N': 0, 'Y': 1}
income_type_map = {'Commercial associate': 0, 'Pensioner': 1, 'State servant': 2, 'Student': 3, 'Working': 4}
education_type_map = {'Academic degree': 0, 'Higher education': 1, 'Incomplete higher': 2, 'Lower secondary': 3, 'Secondary / secondary special': 4}
family_status_map = {'Civil marriage': 0, 'Married': 1, 'Separated': 2, 'Single / not married': 3, 'Widow': 4}
housing_type_map = {'Co-op apartment': 0, 'House / apartment': 1, 'Municipal apartment': 2, 'Office apartment': 3, 'Rented apartment': 4, 'With parents': 5}
occupation_type_map = {
    'Accountants': 0, 'Cleaning staff': 1, 'Cooking staff': 2, 'Core staff': 3, 'Drivers': 4,
    'HR staff': 5, 'High skill tech staff': 6, 'IT staff': 7, 'Laborers': 8, 'Low-skill Laborers': 9,
    'Managers': 10, 'Medicine staff': 11, 'Private service staff': 12, 'Realty agents': 13,
    'Sales staff': 14, 'Secretaries': 15, 'Security staff': 16, 'Waiters/barmen staff': 17, 'Not Specified': 18
}

# --- APP LAYOUT ---
st.title('📈 Credit Approval Prediction App')
st.write("This app predicts whether a credit card application will be approved based on applicant details.")

if model and scaler:
    # --- SIDEBAR FOR USER INPUT ---
    st.sidebar.header('Applicant Information')

    def user_input_features():
        st.sidebar.markdown("### Personal Details")
        code_gender = st.sidebar.selectbox('Gender', list(gender_map.keys()))
        age_years = st.sidebar.slider('Age', 18, 70, 30)
        cnt_children = st.sidebar.slider('Number of Children', 0, 10, 0)
        name_family_status = st.sidebar.selectbox('Family Status', list(family_status_map.keys()))
        cnt_fam_members = st.sidebar.slider('Family Members', 1, 10, 2)
        
        st.sidebar.markdown("### Property & Contact")
        flag_own_car = st.sidebar.selectbox('Owns a Car?', list(yes_no_map.keys()))
        flag_own_realty = st.sidebar.selectbox('Owns Real Estate?', list(yes_no_map.keys()))
        name_housing_type = st.sidebar.selectbox('Housing Type', list(housing_type_map.keys()))
        flag_work_phone = st.sidebar.selectbox('Has a Work Phone?', [0, 1])
        flag_phone = st.sidebar.selectbox('Has a Home Phone?', [0, 1])
        flag_email = st.sidebar.selectbox('Has an Email?', [0, 1])

        st.sidebar.markdown("### Employment & Financial Details")
        amt_income_total = st.sidebar.number_input('Total Annual Income', min_value=25000, max_value=2000000, value=150000, step=1000)
        # Convert DAYS_EMPLOYED from user-friendly "Years Employed" to the negative days format used in training
        years_employed = st.sidebar.slider('Years Employed', 0, 40, 5)
        days_employed = -years_employed * 365
        
        name_income_type = st.sidebar.selectbox('Income Type', list(income_type_map.keys()))
        name_education_type = st.sidebar.selectbox('Education Level', list(education_type_map.keys()))
        occupation_type = st.sidebar.selectbox('Occupation', list(occupation_type_map.keys()))

        # --- PREPROCESSING ---
        # 1. Map categorical inputs to their encoded numbers
        data = {
            'CODE_GENDER': gender_map[code_gender],
            'FLAG_OWN_CAR': yes_no_map[flag_own_car],
            'FLAG_OWN_REALTY': yes_no_map[flag_own_realty],
            'CNT_CHILDREN': cnt_children,
            'AMT_INCOME_TOTAL': amt_income_total,
            'NAME_INCOME_TYPE': income_type_map[name_income_type],
            'NAME_EDUCATION_TYPE': education_type_map[name_education_type],
            'NAME_FAMILY_STATUS': family_status_map[name_family_status],
            'NAME_HOUSING_TYPE': housing_type_map[name_housing_type],
            'DAYS_EMPLOYED': days_employed,
            'FLAG_WORK_PHONE': flag_work_phone,
            'FLAG_PHONE': flag_phone,
            'FLAG_EMAIL': flag_email,
            'OCCUPATION_TYPE': occupation_type_map[occupation_type],
            'CNT_FAM_MEMBERS': float(cnt_fam_members),
            'AGE_YEARS': age_years,
            # Placeholder values for columns not collected from user but present in the model's training data
            'ID': 0, 
            'FLAG_MOBIL': 1, 
            'MONTHS_BALANCE': -15 # Using a common value as a placeholder
        }
        
        features = pd.DataFrame(data, index=[0])

        # 2. Scale the income
        features['AMT_INCOME_TOTAL'] = scaler.transform(features[['AMT_INCOME_TOTAL']])

        # 3. Ensure column order matches the model's training order
        expected_features = [
            'ID', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_EMPLOYED',
            'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
            'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'MONTHS_BALANCE', 'AGE_YEARS'
        ]
        return features[expected_features]

    # Get user input and preprocess it
    input_df = user_input_features()

    # --- PREDICTION AND DISPLAY ---
    st.subheader('Applicant Data Summary')
    st.write("The following data has been processed and will be used for prediction:")
    st.dataframe(input_df)

    if st.button('**Predict Approval Status**', type="primary"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Prediction Result')
       
        if prediction[0] == 1:
            st.success('**Approved** (Low Risk)')
            st.progress(prediction_proba[0][1])
            st.write(f"Confidence: {prediction_proba[0][1]:.2%}")
        else:
            st.error('**Rejected** (High Risk)')
            st.progress(prediction_proba[0][0])
            st.write(f"Confidence: {prediction_proba[0][0]:.2%}")