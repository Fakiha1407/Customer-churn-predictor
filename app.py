
import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title='Customer Churn Predictor', page_icon='📉', layout='centered')
st.title('Customer Churn Predictor')
st.markdown('Enter customer details below to predict churn risk.')
st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Account Info')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless = st.radio('Paperless Billing', ['Yes', 'No'])
    senior = st.radio('Senior Citizen', ['No', 'Yes'])
    partner = st.radio('Has Partner', ['Yes', 'No'])
    dependents = st.radio('Has Dependents', ['Yes', 'No'])

with col2:
    st.subheader('Services & Charges')
    monthly_charges = st.slider('Monthly Charges ($)', 18.0, 120.0, 65.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=500.0)
    internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.radio('Online Security', ['Yes', 'No'])
    tech_support = st.radio('Tech Support', ['Yes', 'No'])
    streaming_tv = st.radio('Streaming TV', ['Yes', 'No'])
    streaming_movies = st.radio('Streaming Movies', ['Yes', 'No'])

st.markdown('---')

if st.button('Predict Churn Risk', use_container_width=True):
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_map = {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}
    internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    yn = lambda x: 1 if x == 'Yes' else 0

    total_services = sum([yn(online_security), 0, 0, yn(tech_support), yn(streaming_tv), yn(streaming_movies)])
    charges_per_tenure = monthly_charges / (tenure + 1)

    input_data = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract_map[contract],
        'InternetService': internet_map[internet],
        'PaymentMethod': payment_map[payment],
        'PaperlessBilling': yn(paperless),
        'OnlineSecurity': yn(online_security),
        'TechSupport': yn(tech_support),
        'StreamingTV': yn(streaming_tv),
        'StreamingMovies': yn(streaming_movies),
        'charges_per_month_tenure': charges_per_tenure,
        'total_services': total_services,
        'SeniorCitizen': yn(senior),
        'Partner': yn(partner),
        'Dependents': yn(dependents)
    }])

    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        tier, color = 'Low Risk', 'green'
        advice = 'This customer is likely to stay. Standard retention measures apply.'
    elif prob < 0.6:
        tier, color = 'Medium Risk', 'orange'
        advice = 'This customer shows some churn signals. Consider a loyalty offer.'
    else:
        tier, color = 'High Risk', 'red'
        advice = 'This customer is likely to churn. Immediate retention action recommended.'

    st.markdown(f'### Churn Probability: **{prob*100:.1f}%**')
    st.markdown(f'### Risk Tier: :{color}[**{tier}**]')
    st.progress(float(prob))
    st.info(advice)

    st.markdown('---')
    st.subheader('Key Factors for This Customer')
    factors = {
        'Contract Type': contract,
        'Tenure': f'{tenure} months',
        'Monthly Charges': f'${monthly_charges:.2f}',
        'Internet Service': internet,
        'Online Security': online_security,
        'Tech Support': tech_support,
        'Payment Method': payment
    }
    for k, v in factors.items():
        st.write(f'**{k}:** {v}')

st.markdown('---')
st.caption('Model: XGBoost | Dataset: IBM Telco Customer Churn | ROC-AUC: 0.836 | Built by Fakiha Balouch')
