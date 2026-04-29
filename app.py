import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# TITLE
st.set_page_config(layout="wide")
st.title("💼 Employee Attrition Prediction Dashboard")

# LOAD DATA
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ---------------- DATASET VIEW ----------------
st.subheader("Dataset Viewer")

if st.checkbox("Show Full Dataset"):
    st.dataframe(df)
else:
    st.dataframe(df.head())

# ---------------- SEARCH ----------------
st.subheader("Search Employee Data")

col_name = st.selectbox("Select Column", df.columns)
search_value = st.text_input("Enter value to search")

if search_value:
    filtered_df = df[df[col_name].astype(str).str.contains(search_value, case=False)]
    st.dataframe(filtered_df)

# ---------------- AUTO FILL ----------------
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

# ---------------- 📊 DASHBOARD ----------------
st.subheader("📊 Dashboard Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    df['Attrition'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title("Attrition Count")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Department', hue='Attrition', data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax3)
    ax3.set_title("Age Distribution")
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax4)
    ax4.set_title("Salary vs Attrition")
    st.pyplot(fig4)

# ---------------- 📌 METRICS ----------------
st.subheader("📌 Key Metrics")

total_emp = len(df)
attrition_rate = (df['Attrition'].value_counts(normalize=True)['Yes']) * 100

colm1, colm2 = st.columns(2)
colm1.metric("Total Employees", total_emp)
colm2.metric("Attrition Rate (%)", round(attrition_rate, 2))

# ---------------- PREPROCESSING ----------------
df_model = df.copy()

encoders = {}
for col in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------- ACCURACY ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {round(acc,2)}")

# ---------------- SIDEBAR INPUT ----------------
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

# ---------------- PREDICTION ----------------
st.subheader("🤖 Prediction")

if st.button("Predict Attrition"):

    # ✅ FIX: use full row data (no missing columns)
    input_data = df_model.iloc[[row_number]].drop("Attrition", axis=1)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    actual = df.iloc[row_number]['Attrition']
    st.write(f"📊 Actual (Dataset): {actual}")

    if prediction == 1:
        st.error("⚠️ Employee will leave")
    else:
        st.success("✅ Employee will stay")

    st.info(f"Confidence: {round(prob,2)}")

    if prob < 0.3:
        st.write("Risk Level: 🟢 Low")
    elif prob < 0.6:
        st.write("Risk Level: 🟡 Medium")
    else:
        st.write("Risk Level: 🔴 High")

# FOOTER
st.markdown("---")
st.write("Developed by Maulik 🚀")