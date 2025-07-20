import streamlit as st
import pandas as pd
import numpy as np
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
    try:
        df = pd.read_csv(uploaded_file)

        # 🧼 Drop rows with missing values
        if df.isnull().sum().sum() > 0:
            st.warning("⚠️ Dataset had missing values. Dropping incomplete rows.")
            df = df.dropna()
        if df.empty:
            st.error("❌ All rows removed due to missing data.")
            st.stop()

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        # 🎯 Auto-detect target column
        if "Class" in df.columns:
            target_column = "Class"
        elif "Pass/Fail" in df.columns:
            target_column = "Pass/Fail"
        else:
            st.error("❌ Could not auto-detect target column. Make sure it has a 'Class' or 'Pass/Fail' column.")
            st.stop()

        # 🔄 Auto-convert target to binary (1 = Pass, 0 = At Risk)
        success_values = ['H', 'High', 'Pass', 'Yes', '1', 1]
        df[target_column] = df[target_column].apply(lambda x: 1 if str(x).strip() in success_values else 0)

        # 🧠 Auto-select relevant features (drop ID, gender, nationality, etc.)
        excluded_cols = ['gender', 'NationalITy', 'PlaceofBirth', target_column]
        feature_columns = [col for col in df.columns if col not in excluded_cols and df[col].dtype in [np.number, "object"]]

        # 👥 Save ID or Name column for display if available
        display_id_col = None
        for col in ['StudentID', 'Name', 'ID']:
            if col in df.columns:
                display_id_col = col
                break

        st.info(f"🎯 Using '{target_column}' as target. Predicting student success...")

        # 🔍 Model selection
        model_name = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
        use_grid = st.checkbox("Use Grid Search for Hyperparameter Tuning?")

        if st.button("🚀 Predict Now"):
            with st.spinner("Training model and making predictions..."):
                try:
                    model, X_test, y_test, best_params, cv_score = train_model(
                        df, feature_columns, target_column, model_name, use_grid
                    )

                    st.success("✅ Prediction complete!")
                    st.write(f"📊 Cross-validated Accuracy: **{cv_score:.4f}**")
                    if best_params:
                        st.write("🔧 Best Parameters:", best_params)

                    # 🔮 Predict and display student outcomes
                    y_pred = model.predict(X_test)
                    outcome_map = {
                        0: "⚠️ At Risk",
                        1: "✅ Likely to Pass"
                    }

                    # Build result table
                    result_table = pd.DataFrame({
                        "Prediction": y_pred,
                        "Status": [outcome_map[p] for p in y_pred]
                    })

                    if display_id_col:
                        result_table[display_id_col] = df.loc[X_test.index, display_id_col].values
                        result_table = result_table[[display_id_col, "Prediction", "Status"]]

                    st.subheader("📋 Student Outcomes")
                    st.dataframe(result_table)

                    # 📊 Confusion Matrix
                    st.subheader("📉 Confusion Matrix")
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                    st.pyplot(fig)

                    # 📌 Feature Importance
                    st.subheader("📌 Feature Importance")
                    imp_df = get_feature_importance(model)
                    st.bar_chart(imp_df.set_index("Feature"))

                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to load CSV: {str(e)}")

else:
    st.info("👈 Upload a dataset to get started.")
