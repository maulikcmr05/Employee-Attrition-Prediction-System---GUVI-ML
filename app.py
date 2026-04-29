import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------------------
# TITLE
# -------------------------------
st.title("Employee Attrition Prediction Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# -------------------------------
# DATASET VIEWER
# -------------------------------
st.subheader("Dataset Viewer")

show_full = st.checkbox("Show Full Dataset")

if show_full:
    st.dataframe(df)
else:
    st.dataframe(df.head())

# -------------------------------
# SEARCH FEATURE
# -------------------------------
st.subheader("Search Employee Data")

col_name = st.selectbox("Select Column", df.columns)
search_value = st.text_input("Enter value to search")

if search_value:
    filtered_df = df[df[col_name].astype(str).str.contains(search_value, case=False)]
    st.write("Filtered Data:")
    st.dataframe(filtered_df)

# -------------------------------
# AUTO-FILL FROM DATASET 🔥
# -------------------------------
st.subheader("Auto Fill from Dataset")

row_number = st.number_input("Select Row Number", 0, len(df)-1, 0)

# Session state init
if "age" not in st.session_state:
    st.session_state.age = 30
    st.session_state.daily_rate = 500
    st.session_state.distance = 5
    st.session_state.job_level = 2
    st.session_state.monthly_income = 5000
    st.session_state.years = 5

if st.button("Load Data"):
    row = df.iloc[row_number]

    st.session_state.age = int(row['Age'])
    st.session_state.daily_rate = int(row['DailyRate'])
    st.session_state.distance = int(row['DistanceFromHome'])
    st.session_state.job_level = int(row['JobLevel'])
    st.session_state.monthly_income = int(row['MonthlyIncome'])
    st.session_state.years = int(row['YearsAtCompany'])

# -------------------------------
# GRAPH
# -------------------------------
st.subheader("Attrition Count")

attrition = df['Attrition'].value_counts()

fig, ax = plt.subplots()
ax.bar(attrition.index, attrition.values)
ax.set_title("Employee Attrition Count")
ax.set_xlabel("Attrition")
ax.set_ylabel("Count")

st.pyplot(fig)

# -------------------------------
# DATA PREPROCESSING
# -------------------------------
df_model = df.copy()

le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------------------------------
# SIDEBAR INPUT (AUTO FILLED 🔥)
# -------------------------------
st.sidebar.title("Employee Details")

age = st.sidebar.slider("Age", 18, 60, st.session_state.age)
daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, st.session_state.daily_rate)
distance = st.sidebar.slider("Distance From Home", 1, 30, st.session_state.distance)
job_level = st.sidebar.slider("Job Level", 1, 5, st.session_state.job_level)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, st.session_state.monthly_income)
years_at_company = st.sidebar.slider("Years At Company", 0, 40, st.session_state.years)

# Update session
st.session_state.age = age
st.session_state.daily_rate = daily_rate
st.session_state.distance = distance
st.session_state.job_level = job_level
st.session_state.monthly_income = monthly_income
st.session_state.years = years_at_company

# -------------------------------
# PREDICTION
# -------------------------------
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

    # Fill missing columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee will stay")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.write("Developed by Maulik 🚀")