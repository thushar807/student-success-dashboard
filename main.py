import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from modeling import train_model, get_feature_importance

# 🎨 UI Setup
st.set_page_config(page_title="🎓 Student Success Predictor", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: #f9fcff;
            padding: 1rem;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🎓 Student Success Predictor Dashboard</h1>", unsafe_allow_html=True)

# 📂 Upload Data
st.sidebar.markdown("### 📁 Upload Student Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ➕ Feature Selection
    columns = df.columns.tolist()
    target_column = st.selectbox("🎯 Select the target column (Pass/Fail):", columns)
    feature_columns = st.multiselect("🧠 Select features for prediction:", [col for col in columns if col != target_column])

    if feature_columns:
        # ⚙️ Model Selection
        model_name = st.selectbox("🔍 Choose Model", ["Random Forest", "Logistic Regression"])
        use_grid = st.checkbox("Tune Hyperparameters (Grid Search)?")

        if st.button("🚀 Train Model"):
            with st.spinner("Training in progress..."):
                model, X_test, y_test, best_params, cv_score = train_model(df, feature_columns, target_column, model_name, use_grid)

                st.success("✅ Model trained successfully!")
                st.write(f"📊 Cross-validated Accuracy: **{cv_score:.4f}**")
                if best_params:
                    st.write("🔧 Best Parameters:", best_params)

                # 📈 Evaluation
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                st.subheader("📊 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                st.subheader("📉 Confusion Matrix")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                st.pyplot(fig)

                st.subheader("📌 Feature Importance")
                imp_df = get_feature_importance(model)
                st.bar_chart(imp_df.set_index("Feature"))
else:
    st.info("👈 Upload a dataset to get started.")
