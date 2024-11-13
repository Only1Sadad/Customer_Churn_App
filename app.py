import os
import pickle
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import utils as ut

load_dotenv()

# Check if the environment variable is loaded
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {api_key}")  # This should print your API key

if not api_key:
    raise ValueError("API key not found. Please check the .env file.")



# Set the OpenAI API key from environment variable
client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"))

# Load models using a helper function
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
knn_model = load_model('knn_model.pkl')
decision_tree_classifier_model = load_model('dt_model.pkl')
support_vector_model = load_model('svm_model.pkl')
feature_engineering = load_model('xgboost-FeatureEngineered.pkl')
synthetic_minority_ot = load_model('xgboost-SMOTE.pkl')





# Prepare input data for model prediction
def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_of_products, has_credit_card, is_active_member,
                  estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


# Generate model predictions
def make_predictions(input_df, input_dict, df, customer_id):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
        'svm_model' : support_vector_model.predict_proba(input_df)[0][1],
         }
    avg_probability = np.mean(list(probabilities.values()))

    # Calculate feature importances
    feature_importances = xgboost_model.feature_importances_
    feature_names = input_df.columns

    fig_feature_importance = ut.create_feature_importance_chart(
        feature_importances, feature_names)

    # Display gauge chart and model probabilities
    cols1, cols2 = st.columns(2)
    with cols1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a: {avg_probability:.2%} probability of churning."
        )

        # Calculate and display the estimated lifetime customer value
        estimated_lifetime_customer_value = ut.calculate_estimated_lifetime_customer_value(
            df, customer_id)
        st.write(
            f"The estimated lifetime customer value for this customer is: {estimated_lifetime_customer_value:.2f}"
        )

        st.plotly_chart(fig_feature_importance, use_container_width=True)

    with cols2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    st.markdown("### Model Predictions")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob:.2%}")
    st.write(f"Average Probability: {avg_probability:.2%}")

    return avg_probability


# Generate an explanation of the prediction
def explain_prediction(probability, input_dict, surname, df):
    prompt = f"""
    As a bank data scientist, you are interpreting a prediction for a customer named {surname}.
    The model has determined a {round(probability * 100, 1)}% chance of churning based on these key details:

    {input_dict}

    Top model features impacting churn risk:
       Feature              | Importance
       ----------------------|-----------
       NumOfProducts         | 0.323888
       IsActiveMember        | 0.164146
       Age                   | 0.109550
       Geography_Germany     | 0.091373
       Balance               | 0.052786
       Geography_France      | 0.046463
       Gender_Female         | 0.045283
       Geography_Spain       | 0.036855
       CreditScore           | 0.035005
       EstimatedSalary       | 0.032655
       HasCrCard             | 0.031940
       Tenure                | 0.030054

    Statistical summaries:
    - Churned customers: {df[df['Exited'] == 1].describe()}
    - Non-churned customers: {df[df['Exited'] == 0].describe()}

    Generate a 3-sentence explanation based on these features:
    - If the customer has over 40% churn risk, explain why they are at risk.
    - If under 40%, explain why they are likely to stay.
    """
    # Call API with prompt
    raw_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


# Generate a retention email based on churn risk
def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
    You are a customer retention manager at Bank of America. Your role is to create personalized and encouraging emails to retain customers who might be at risk of leaving the bank.

    A customer named {surname} has been identified as potentially at risk of churning. Here is the customers key information:
    {input_dict}

    Based on their information, here is why they might be considering leaving:
    {explanation}

    Please write an engaging email to the customer with the following guidelines:
    - Use a friendly and appreciative tone, thanking them for their loyalty.
    - Offer a list of tailored incentives (e.g., fee waivers, increased rewards) in bullet points, explaining each benefit briefly and positively.
    - Emphasize the value of staying with Bank of America, including benefits such as our online banking platform, financial guidance, and customer support.
    - Avoid mentioning any probabilities, risk scores, or machine learning models to the customer.
    - Conclude with an invitation for the customer to reach out with questions or for further assistance.

    Your email should be professional, clear, and welcoming. Keep the tone friendly and highlight the benefits of staying with Bank of America.
    """
    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


# Streamlit interface
st.title("Customer Churn Prediction")

# Load customer data
df = pd.read_csv("churn.csv")
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]
    selected_customer = df.loc[df['CustomerId'] ==
                               selected_customer_id].iloc[0]

    # Collect user inputs
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))
        location = st.selectbox("Location", ["France", "Germany", "Spain"])
        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == 'Male' else 1)
        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer['Age']))
        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))
        num_of_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer['HasCrCard']))
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']))
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_of_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)
    avg_probability = make_predictions(input_df, input_dict, df,
                                       selected_customer_id)

    explanation = explain_prediction(avg_probability, input_dict,
                                     selected_surname, df)
    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    # Generate and display the email
    email_content = generate_email(avg_probability, input_dict, explanation,
                                   selected_surname)
    st.markdown("---")
    st.subheader("Email to Customer")
    st.markdown(email_content)
