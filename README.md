# Customer Churn Prediction App

## Project Overview

This project provides a customer churn prediction solution using machine learning models. The app predicts the likelihood of a customer churning and provides explanations along with recommendations for customer retention.

### Key Features:
- **Churn Probability Prediction**: Predicts the likelihood of a customer churning based on their demographic and financial data.
- **Model Comparison**: Compares predictions from multiple models (XGBoost, Random Forest, K-Nearest Neighbors, Support Vector Machine).
- **Feature Importance Chart**: Visualizes the relative importance of each feature in making the churn prediction.
- **Churn Prediction Explanation**: Provides a detailed explanation of the churn prediction, highlighting which features were most influential.
- **Personalized Email Generation**: Generates a personalized email for retention purposes based on the churn prediction.

## Technology Stack:
- **Python 3.x**: The project is built using Python.
- **Streamlit**: For building the interactive web application interface.
- **Scikit-Learn**: For machine learning models (e.g., Random Forest, KNN).
- **XGBoost**: For XGBoost classification model.
- **OpenAI API**: Used to generate explanations and personalized emails based on churn predictions.

## Project Structure:
- **App.py**: The main script that runs the Streamlit app.
- **Models/**: Folder containing pre-trained machine learning models (XGBoost, Random Forest, KNN, SVM, etc.).
- **Data/**: Folder containing the dataset (`churn.csv`).
- **utils.py**: Utility functions for feature importance chart generation, customer lifetime value calculation, etc.
- **.env**: Environment variables file for sensitive information like API keys.

## Setup Instructions:

### Prerequisites:
- **Python** 3.12 or above
- Install **pip** for package management

### Steps to Run the App:
1. Clone the repository to your local machine:
   ```bash
   git clone <repository_url>

Set up a virtual environment:
python3 -m venv .venv
source .venv/bin/activate  # On MacOS/Linux
.venv\Scripts\activate     # On Windows


