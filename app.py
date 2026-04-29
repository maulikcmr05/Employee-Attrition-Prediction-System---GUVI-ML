import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---------------- PAGE ----------------
st.set_page_config(layout="wide")
st.title("💼 Employee Attrition Prediction Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ---------------- DATASET VIEW ----------------
st.subheader("📋 Dataset Viewer")

if st.checkbox("Show Full Dataset"):
    st.dataframe(df)   # FULL DATASET
else:
    st.dataframe(df.head())

# ---------------- SEARCH ----------------
st.subheader("🔍 Search Employee Data")

col_name = st.selectbox("Select Column", df.columns)
search_value = st.text_input("Enter value to search")

if search_value:
    filtered_df = df[df[col_name].astype(str).str.contains(search_value, case=False)]
    st.dataframe(filtered_df)

# ---------------- SELECT ROW ----------------
st.subheader("📌 Select Dataset Row")
row_number = st.number_input("Row Number", 0, len(df)-1, 0)

# ---------------- DASHBOARD ----------------
st.subheader("📊 Dashboard")

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

# ---------------- ENCODING ----------------
df_model = df.copy()

for col in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ---------------- PREDICTION (DATASET ROW) ----------------
st.subheader("🤖 Prediction (Dataset Row)")

if st.button("Predict Selected Row"):
    input_data = X.iloc[[row_number]]

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    actual = df.iloc[row_number]['Attrition']
    st.write(f"📊 Actual: {actual}")

    if prediction == 1:
        st.error("⚠️ Employee will leave")
    else:
        st.success("✅ Employee will stay")

    st.info(f"Confidence: {round(prob,2)}")

# ---------------- MANUAL INPUT ----------------
st.sidebar.title("🧑‍💻 Manual Input")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance From Home", 1, 30, 5)

if st.sidebar.button("Predict Manual"):

    input_manual = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

    input_manual['Age'] = age
    input_manual['MonthlyIncome'] = income
    input_manual['DistanceFromHome'] = distance

    prediction = model.predict(input_manual)[0]
    prob = model.predict_proba(input_manual)[0][1]

    if prediction == 1:
        st.sidebar.error("⚠️ Will Leave")
    else:
        st.sidebar.success("✅ Will Stay")

    st.sidebar.write(f"Confidence: {round(prob,2)}")

# ---------------- METRICS ----------------
st.subheader("📌 Key Metrics")

total = len(df)
attrition_rate = df['Attrition'].value_counts(normalize=True)['Yes'] * 100

col1, col2 = st.columns(2)
col1.metric("Total Employees", total)
col2.metric("Attrition Rate (%)", round(attrition_rate, 2))

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Developed by Maulik 🚀")