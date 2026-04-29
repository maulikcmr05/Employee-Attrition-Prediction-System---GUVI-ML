import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ---------------- PAGE ----------------
st.set_page_config(layout="wide")
st.title("💼 Employee Attrition Prediction Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ---------------- VIEW ----------------
st.subheader("Dataset Viewer")
st.dataframe(df.head())

# ---------------- SEARCH ----------------
st.subheader("Search Employee Data")

col_name = st.selectbox("Select Column", df.columns)
search_value = st.text_input("Enter value to search")

if search_value:
    filtered_df = df[df[col_name].astype(str).str.contains(search_value, case=False)]
    st.dataframe(filtered_df)

# ---------------- ENCODING ----------------
df_model = df.copy()
encoders = {}

for col in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------- PERFORMANCE ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("Accuracy", round(acc, 2))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
st.pyplot(fig_cm)

# ---------------- DASHBOARD ----------------
st.subheader("📊 Dashboard")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    df['Attrition'].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Department', hue='Attrition', data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ---------------- AUTO LOAD ----------------
st.subheader("Auto Fill from Dataset")

row_number = st.number_input("Select Row Number", 0, len(df)-1, 0)

# ---------------- PREDICTION ----------------
st.subheader("🤖 Prediction")

if st.button("Predict using Dataset Row"):

    # 🔥 IMPORTANT FIX (FULL ROW USED)
    input_data = df_model.iloc[[row_number]].drop("Attrition", axis=1)

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
st.sidebar.title("Manual Input")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance From Home", 1, 30, 5)

if st.sidebar.button("Predict Manual"):

    # Create empty input with all features
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

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("Developed by Maulik 🚀")