import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Feature importance chart for Churn Prediction
def create_feature_importance_chart(feature_importances, feature_names):
    # Define the list of features you want to show
    selected_features = ["NumOfProducts", "Balance",
                         "EstimatedSalary", "Tenure", "CreditScore"]

    # Create a DataFrame from the feature importances and feature names
    feature_df = pd.DataFrame({
     'Feature': feature_names,
    'Importance': feature_importances
})
    # Filter the DataFrame to include only the selected features
    feature_df = feature_df[feature_df['Feature'].isin(selected_features)]
    
    # Sort the DataFrame by importance 
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    
    # Create a horizontal bar chart using plotly express
    fig = px.bar(
        feature_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Customer Percentile"
    )
    # Update x-axis to show percentages and change axis label to "Percentile"
    fig.update_layout(
        xaxis_tickformat=".0%", 
        xaxis_title="Percentile", 
        yaxis_title="Metric"
    )

     # Return the figure
    return fig

# Gauge chart for Churn Probability
def create_gauge_chart(probability):
    # Determine color based on churn probability
    if probability < 0.3:
        color = "green"
    elif probability < 0.6:
        color = "yellow"
    else:
        color = "red"

    # Create a gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Churn Probability', 'font': {'size': 24, 'color': 'white'}},
            number={'font': {'size': 40, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'white'},
                'bar': {'color': color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                    {'range': [30, 60], 'color': "rgba(255, 255, 0, 0.3)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        )
    )

    # Update chart layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Return the chart
    return fig

# Model probability bar chart
def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(data=[
        go.Bar(
            y=models,
            x=probs,
            orientation='h',
            text=[f'{p:.2%}' for p in probs],
            textposition='auto'
        )
    ])

    # Update layout for better appearance
    fig.update_layout(
        title='Churn Probability by Model',
        yaxis_title='Models',
        xaxis_title='Probability',
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

# Calculate estimated lifetime customer value
def calculate_estimated_lifetime_customer_value(df, customer_id, churn_rate=0.1):
    customer = df.loc[df['CustomerId'] == customer_id]

    balance = customer['Balance'].iloc[0]
    num_of_products = customer['NumOfProducts'].iloc[0]
    has_credit_card = customer['HasCrCard'].iloc[0]
    is_active_member = customer['IsActiveMember'].iloc[0]
    estimated_salary = customer['EstimatedSalary'].iloc[0]

    monthly_income = estimated_salary / 12
    monthly_credit_card_fee = 0.01 * balance if has_credit_card else 0
    monthly_active_member_fee = 0.01 * balance if is_active_member else 0
    monthly_churn_fee = churn_rate * balance

    lifetime_customer_value = (monthly_income + monthly_credit_card_fee +
                               monthly_active_member_fee - monthly_churn_fee)

    return lifetime_customer_value
