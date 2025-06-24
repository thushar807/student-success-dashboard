import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification\_report
from sklearn.preprocessing import LabelEncoder

st.set\_page\_config(page\_title="Student Success Predictor", layout="wide")
st.title("📊 Student Success Predictor Dashboard")

# File upload

uploaded\_file = st.file\_uploader("Upload a student dataset (CSV format)", type=\["csv"])

if uploaded\_file:
df = pd.read\_csv(uploaded\_file)
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

```
# Automatically detect categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.subheader("⚙️ Model Settings")
target_column = st.selectbox("Select the target column (what to predict):", options=cat_cols)
feature_columns = st.multiselect("Select input features:", options=[col for col in df.columns if col != target_column])

if target_column and feature_columns:
    data = df[feature_columns + [target_column]].dropna()

    # Encode categorical columns
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le

    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully!")

    # Input form for prediction
    st.subheader("📥 Enter Student Details for Prediction")
    user_input = {}
    for col in feature_columns:
        if col in encoders:
            user_input[col] = st.selectbox(f"{col}", encoders[col].classes_)
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            user_input[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=min_val)

    # Convert user input to dataframe
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]

    target_labels = encoders[target_column].classes_ if target_column in encoders else sorted(y.unique())

    st.subheader("🔮 Prediction Result")
    st.write(f"**Predicted Class:** {encoders[target_column].inverse_transform([prediction])[0] if target_column in encoders else prediction}")

    st.subheader("📈 Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(target_labels, pred_proba)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    st.pyplot(fig)

    st.subheader("💡 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

    st.subheader("📊 Model Evaluation (on test set)")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
```

else:
st.warning("Please upload a dataset to continue.")
