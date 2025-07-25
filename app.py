import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Classifier", layout="centered")

# Load the full pipeline and SHAP explainer
classifier = joblib.load("models/threshold_tuned_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Set threshold (same as in train_model)
BEST_THRESHOLD = 0.4139  # update this if it changes in train_model

st.title("Credit Risk Scoring App")
st.markdown("This app predicts whether a customer is likely to default on credit using a trained Gradient Boosting model.")

# --- User Inputs ---
st.header("Customer Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.selectbox("Job (0 = unskilled, 3 = highly skilled)", [0, 1, 2, 3])
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    saving_accounts = st.selectbox("Saving Accounts", ["NA", "little", "moderate", "rich"])

with col2:
    checking_account = st.selectbox("Checking Account", ["NA", "little", "moderate", "rich"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    duration = st.number_input("Duration (months)", min_value=1, value=12)
    purpose = st.selectbox("Purpose", [
        "radio/TV", "education", "furniture/equipment", "car",
        "business", "domestic appliances", "repairs", "vacation/others"
    ])

# --- Predict ---
if st.button("Predict Credit Risk"):
    # 1. Format input
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])

# Transform input
    try:
        X_transformed = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # 2. Predict probability

    probability = classifier.predict_proba(X_transformed)[0][1]
    prediction = classifier.predict(X_transformed)[0]

    st.write(f"Risk Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
    st.write(f"Probability of Risk: {probability:.2f}")

    # SHAP Explanation
    if explainer:
        shap_values = explainer(X_transformed)

        st.subheader("SHAP Explanation")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
