import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.title("Employee Attrition Prediction Dashboard")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Attrition Count")

attrition = df['Attrition'].value_counts()

fig, ax = plt.subplots()
ax.bar(attrition.index, attrition.values)
ax.set_title("Employee Attrition Count")
ax.set_xlabel("Attrition")
ax.set_ylabel("Count")

st.pyplot(fig)

df_model = df.copy()

le = LabelEncoder()

for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

# Split data
X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.sidebar.title("Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, 500)
distance = st.sidebar.slider("Distance From Home", 1, 30, 5)
job_level = st.sidebar.slider("Job Level", 1, 5, 2)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
years_at_company = st.sidebar.slider("Years At Company", 0, 40, 5)

st.subheader("Prediction")

if st.button("Predict Attrition"):
    input_data = pd.DataFrame({
        'Age': [age],
        'DailyRate': [daily_rate],
        'DistanceFromHome': [distance],
        'JobLevel': [job_level],
        'MonthlyIncome': [monthly_income],
        'YearsAtCompany': [years_at_company]
    })

    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee will stay")

st.markdown("---")
st.write("Developed by Maulik 🚀")