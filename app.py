import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Attrition Dashboard", layout="wide")

st.title("💼 Employee Attrition Prediction Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["Manual Prediction", "Dataset Prediction", "Insights"])

# =====================================================
# 🟢 1. MANUAL PREDICTION
# =====================================================
if mode == "Manual Prediction":
    st.header("📝 Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        distance = st.slider("Distance From Home", 1, 30, 10)

    with col2:
        income = st.number_input("Monthly Income", 1000, 20000, 5000)
        overtime = st.selectbox("OverTime", ["Yes", "No"])

    overtime_val = 1 if overtime == "Yes" else 0

    if st.button("Predict Attrition"):
        data = np.array([[age, income, distance, overtime_val]])

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        risk = "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"

        st.success("Prediction: " + ("Will Leave ❌" if prediction==1 else "Will Stay ✅"))
        st.info(f"Probability: {round(prob,2)}")
        st.warning(f"Risk Level: {risk}")

# =====================================================
# 🔵 2. DATASET PREDICTION
# =====================================================
elif mode == "Dataset Prediction":
    st.header("📊 Select Employee from Dataset")

    st.dataframe(df, use_container_width=True)

    index = st.number_input("Enter Row Number", 0, len(df)-1, 0)

    if st.button("Predict Selected Employee"):
        row = df.iloc[index]

        data = np.array([[
            row['Age'],
            row['MonthlyIncome'],
            row['DistanceFromHome'],
            1 if row['OverTime'] == "Yes" else 0
        ]])

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        risk = "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"

        st.success("Prediction: " + ("Will Leave ❌" if prediction==1 else "Will Stay ✅"))
        st.info(f"Probability: {round(prob,2)}")
        st.warning(f"Risk Level: {risk}")

# =====================================================
# 🟡 3. INSIGHTS
# =====================================================
else:
    st.header("📈 Insights & Analysis")

    st.subheader("Key Insights")
    st.write("• Employees working overtime are more likely to leave")
    st.write("• Low income increases attrition risk")
    st.write("• Long distance from home affects retention")
    st.write("• Younger employees tend to leave more")

    st.subheader("Recommendations")
    st.write("✔ Improve salary structure")
    st.write("✔ Reduce overtime workload")
    st.write("✔ Improve work-life balance")
    st.write("✔ Provide career growth opportunities")