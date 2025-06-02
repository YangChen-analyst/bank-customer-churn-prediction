import joblib
import numpy as np

model = joblib.load("xgboost_churn_model.pkl")
scaler = joblib.load("standard_scaler.pkl")

def predict_customer(features):
    """
    Predict churn for a new customer.

    Args:
    input_features_list: list of numeric features in the same order as model expects

    Returns:
    prediction: 0 (not churn) or 1 (churn)
    """
    X_scaled = scaler.transform([features])
    return model.predict(X_scaled)

# Example customer: 
# credit_score, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, age_flag,
# high_balance, country_Germany, country_Spain, gender_Male, roducts_grouped_2, poducts_grouped_3+
customer = [650, 45, 1, 20000, 1, 1, 1, 75000, 0, 0, False, False, True, False, False]  # → match feature order
result = predict_customer(customer)
print("Churn Prediction:", "Yes" if result == 1 else "No")