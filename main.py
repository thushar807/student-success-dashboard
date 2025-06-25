import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, RocCurveDisplay, PrecisionRecallDisplay
from modeling import train_model, get_feature_importance

st.set_page_config(page_title="🎓 Student Success Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🎓 Student Success Predictor Dashboard</h1>", unsafe_allow_html=True)

# Upload and read dataset
with st.sidebar:
    st.markdown("### 📁 Upload Student Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("⚙️ Model Configuration")
    target_column = st.selectbox("🎯 Select the target column (e.g., Pass/Fail):", df.columns)
    feature_columns = st.multiselect("🧠 Select input features: ", [col for col in df.columns if col != target_column])
    model_name = st.selectbox("🔍 Choose model", ["Random Forest", "Logistic Regression"])
    tune = st.checkbox("🔧 Use Grid Search for hyperparameter tuning")

    if feature_columns and target_column:
        model, X_test, y_test, best_params, cv_score = train_model(df, feature_columns, target_column, model_name, grid_search=tune)

        st.success("✅ Model trained successfully!")
        if best_params:
            st.write("Best parameters:", best_params)
        if cv_score:
            st.write(f"Cross-validated Accuracy: {cv_score:.4f}")

        st.subheader("📈 Evaluation Results")

        # Confusion Matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Feature Importance
        st.subheader("🔍 Feature Importance")
        importance_df = get_feature_importance(model)
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

        # ROC / PR Curves
        try:
            st.subheader("📊 ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
            st.pyplot(fig_roc)

            st.subheader("📊 Precision-Recall Curve")
            fig_pr, ax_pr = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr)
            st.pyplot(fig_pr)
        except Exception as e:
            st.warning("ROC/PR curves not available for this model.")

else:
    st.warning("👈 Please upload a CSV file to begin.")
