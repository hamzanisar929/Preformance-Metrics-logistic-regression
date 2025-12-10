import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

@st.cache_data
def train_model():
    iris = pd.read_csv("IRIS.csv")

    X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = iris["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred)

    return log_reg, accuracy, precision, recall, f1, report


model, accuracy, precision, recall, f1, report = train_model()


st.title("🌸 Iris Species Predictor (Logistic Regression)")
st.write(
    """
This app uses **Logistic Regression** to classify Iris flowers  
based on sepal and petal measurements.  
"""
)

st.header("🔢 Enter Flower Measurements")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)

if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"🌼 Predicted Species: **{prediction}**")

st.header("📊 Model Evaluation Metrics")
st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")

st.subheader("📄 Classification Report")
st.text(report)
