
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Student Success Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Success Predictor Dashboard</h1>", unsafe_allow_html=True)

# Upload section
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a student dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Attempt to detect numeric grade column
    grade_cols = [col for col in df.columns if 'G' in col.upper() and df[col].dtype in [np.int64, np.float64]]
    if not grade_cols:
        st.error("No grade column detected for Pass/Fail conversion. Please ensure dataset has final grades.")
    else:
        final_grade_col = grade_cols[-1]
        df['Success'] = df[final_grade_col].apply(lambda x: 1 if x >= 50 else 0)

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("⚙️ Model Settings")
        feature_options = [col for col in df.columns if col not in ['Success']]
        feature_columns = st.multiselect("Select input features:", options=feature_options)

        if feature_columns:
            data = df[feature_columns + ['Success']].dropna()

            # Encode categorical features
            encoders = {}
            for col in data.columns:
                if data[col].dtype == 'object':
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    encoders[col] = le

            X = data[feature_columns]
            y = data['Success']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            st.success("✅ Model trained successfully!")

            # User prediction
            st.subheader("📥 Enter Student Details for Prediction")
            user_input = {}
            for col in feature_columns:
                if col in encoders:
                    user_input[col] = st.selectbox(f"{col}", encoders[col].classes_)
                else:
                    user_input[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_df = pd.DataFrame([user_input])
            for col in input_df.columns:
                if col in encoders:
                    input_df[col] = encoders[col].transform(input_df[col])

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            # 📌 Prediction Result
            st.subheader("🎯 Prediction Result")
            prediction_label = "Pass" if pred == 1 else "Fail"
            st.metric(label="🎓 Predicted Outcome", value=prediction_label)

            # Probability bar chart
            st.subheader("📊 Prediction Probabilities")
            fig, ax = plt.subplots()
            ax.bar(['Fail', 'Pass'], proba, color=['red', 'green'])
            ax.set_ylabel("Probability")
            st.pyplot(fig)

            # Feature importance
            st.subheader("📌 Feature Importance")
            importances = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.bar_chart(importances.set_index("Feature"))

            # Report summary
            st.subheader("📈 Model Evaluation (F1 Score & Accuracy)")
            report = classification_report(y_test, y_pred, output_dict=True)
            summary = pd.DataFrame({
                "Accuracy": [report["accuracy"]],
                "F1 Score": [report["1"]["f1-score"]],
                "Precision": [report["1"]["precision"]],
                "Recall": [report["1"]["recall"]]
            }).T.rename(columns={0: "Score"})
            st.dataframe(summary.style.highlight_max(axis=0, color="lightgreen"))

else:
    st.info("👈 Please upload a dataset from the sidebar to begin.")
