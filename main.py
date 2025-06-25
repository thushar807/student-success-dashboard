import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from modeling import train_model, get_feature_importance

# Page Config
st.set_page_config(page_title="🎓 Student Success Predictor", layout="wide")
st.markdown("""
    <style>
    body {
        background-image: url("https://raw.githubusercontent.com/ThusharStorage/student-assets/main/wow-office.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(6px);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🎓 Student Success Predictor Dashboard</h1>", unsafe_allow_html=True)

# Upload Data
with st.sidebar:
    st.header("📁 Upload Student Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Create PassStatus if missing
    if "PassStatus" not in df.columns:
        grade_cols = ["G3", "final_grade", "FinalGrade", "grade"]
        found = False
        for col in grade_cols:
            if col in df.columns:
                df["PassStatus"] = df[col].apply(lambda x: "Pass" if x >= 10 else "Fail")
                found = True
                break
        if not found:
            st.error("❌ Could not infer Pass/Fail column. Please include a grade column.")
            st.stop()

    st.subheader("⚙️ Model Configuration")
    target_column = "PassStatus"
    feature_columns = st.multiselect("🧠 Select features for prediction:", options=[col for col in df.columns if col != target_column])
    model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    use_grid = st.checkbox("Enable Hyperparameter Tuning (GridSearchCV)")

    if feature_columns:
        with st.spinner("Training model..."):
            model, X_test, y_test, best_params, cv_score = train_model(
                df=df,
                feature_columns=feature_columns,
                target_column=target_column,
                model_name=model_type,
                grid_search=use_grid
            )

        st.success("✅ Model trained successfully!")

        if best_params:
            st.info(f"🔍 Best Parameters: {best_params}")
        st.write(f"📈 Cross-validated Score: {cv_score:.4f}")

        # Prediction Form
        st.subheader("📝 Enter Student Details")
        user_input = {}
        for col in feature_columns:
            if df[col].dtype == "object":
                user_input[col] = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
            else:
                user_input[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("🎯 Prediction Result")
        st.success(f"Predicted Outcome: {prediction}")

        st.subheader("📊 Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(model.classes_, proba, color="#1f77b4")
        ax.set_ylabel("Probability")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)

        st.subheader("💡 Feature Importance")
        imp_df = get_feature_importance(model)
        st.bar_chart(imp_df.set_index("Feature"))

        st.subheader("📈 Model Evaluation")
        y_pred = model.predict(X_test)
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report)

        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
        ax2.set_title("Confusion Matrix")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

else:
    st.warning("👈 Upload a CSV file to get started.")
