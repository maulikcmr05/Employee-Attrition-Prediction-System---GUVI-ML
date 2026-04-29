import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# TITLE
st.title("💼 Employee Attrition Prediction Dashboard")

# LOAD DATA
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# DATASET VIEWER
st.subheader("Dataset Viewer")

if st.checkbox("Show Full Dataset"):
    st.dataframe(df)
else:
    st.dataframe(df.head())

# SEARCH
st.subheader("Search Employee Data")

col_name = st.selectbox("Select Column", df.columns)
search_value = st.text_input("Enter value to search")

if search_value:
    filtered_df = df[df[col_name].astype(str).str.contains(search_value, case=False)]
    st.dataframe(filtered_df)

# AUTO FILL
st.subheader("Auto Fill from Dataset")

row_number = st.number_input("Select Row Number", 0, len(df)-1, 0)

if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "Age": 30,
        "DailyRate": 500,
        "DistanceFromHome": 5,
        "JobLevel": 2,
        "MonthlyIncome": 5000,
        "YearsAtCompany": 5
    }

if st.button("Load Data"):
    row = df.iloc[row_number]
    st.session_state.inputs = {
        "Age": int(row['Age']),
        "DailyRate": int(row['DailyRate']),
        "DistanceFromHome": int(row['DistanceFromHome']),
        "JobLevel": int(row['JobLevel']),
        "MonthlyIncome": int(row['MonthlyIncome']),
        "YearsAtCompany": int(row['YearsAtCompany'])
    }

# GRAPH
st.subheader("Attrition Count")

attrition = df['Attrition'].value_counts()

fig, ax = plt.subplots()
ax.bar(attrition.index, attrition.values)
ax.set_title("Employee Attrition Count")

st.pyplot(fig)

# PREPROCESSING
df_model = df.copy()

encoders = {}
for col in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

# MODEL
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# SIDEBAR INPUT
st.sidebar.title("Employee Details")

inputs = st.session_state.inputs

age = st.sidebar.slider("Age", 18, 60, inputs["Age"])
daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, inputs["DailyRate"])
distance = st.sidebar.slider("Distance From Home", 1, 30, inputs["DistanceFromHome"])
job_level = st.sidebar.slider("Job Level", 1, 5, inputs["JobLevel"])
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, inputs["MonthlyIncome"])
years = st.sidebar.slider("Years At Company", 0, 40, inputs["YearsAtCompany"])

st.session_state.inputs = {
    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance,
    "JobLevel": job_level,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years
}

# PREDICTION
st.subheader("Prediction")

if st.button("Predict Attrition"):

    input_data = pd.DataFrame([st.session_state.inputs])

    # Fill missing columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    actual = df.iloc[row_number]['Attrition']

    st.write(f"📊 Actual (Dataset): {actual}")

    if prediction == 1:
        st.error("⚠️ Predicted: Employee will leave")
    else:
        st.success("✅ Predicted: Employee will stay")

    st.info(f"Confidence: {round(prob,2)}")

    # ---------------- RISK LEVEL 🔥 ----------------
    if prob < 0.3:
        risk = "Low Risk 🟢"
    elif prob < 0.6:
        risk = "Medium Risk 🟡"
    else:
        risk = "High Risk 🔴"

    st.write("Risk Level:", risk)

# ACCURACY
from sklearn.metrics import accuracy_score

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

st.write(f"Model Accuracy: {round(acc,2)}")

# FOOTER
st.markdown("---")
st.write("Developed by Maulik 🚀")