import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="🎓 Student Success Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: green;'>🎓 Student Success Predictor Dashboard</h1>", unsafe_allow_html=True)

# Upload section
with st.sidebar:
    st.subheader("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload a student dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --- STEP 1: Create 'PassStatus' if not present ---
    if "PassStatus" not in df.columns:
        # Try to infer from final grade
        possible_grade_cols = ["G3", "final_grade", "FinalGrade", "grade"]
        found = False
        for col in possible_grade_cols:
            if col in df.columns:
                df["PassStatus"] = df[col].apply(lambda x: "Pass" if x >= 10 else "Fail")
                found = True
                break

        if not found:
            # Try behavioral proxy
            behavior_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
            available = [col for col in behavior_cols if col in df.columns]
            if available:
                df["behavior_score"] = df[available].sum(axis=1)
                threshold = df["behavior_score"].mean()
                df["PassStatus"] = df["behavior_score"].apply(lambda x: "Pass" if x >= threshold else "Fail")
                df.drop(columns=["behavior_score"], inplace=True)
                st.info("Pass/Fail status inferred using behavior-based proxy.")
            else:
                st.error("❌ Could not detect or infer a 'Pass/Fail' outcome. Please include a grade column.")
                st.stop()

    # --- STEP 2: Model setup ---
    st.subheader("⚙️ Model Settings")
    target_column = "PassStatus"
    feature_columns = st.multiselect("Select input features for prediction:", options=[col for col in df.columns if col != target_column])

    if feature_columns:
        data = df[feature_columns + [target_column]].dropna()
        encoders = {}

        for col in data.columns:
            if data[col].dtype == "object":
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                encoders[col] = le

        X = data[feature_columns]
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        # --- STEP 3: Input form ---
        st.subheader("📝 Enter Student Details for Prediction")
        user_input = {}
        for col in feature_columns:
            if col in encoders:
                user_input[col] = st.selectbox(f"{col}", encoders[col].classes_)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                user_input[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=min_val)

        input_df = pd.DataFrame([user_input])
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        label_names = encoders[target_column].classes_ if target_column in encoders else sorted(y.unique())

        # --- STEP 4: Results ---
        st.subheader("🎯 Prediction Result")
        st.write(f"**Predicted Class:** {encoders[target_column].inverse_transform([prediction])[0] if target_column in encoders else prediction}")

        st.subheader("📊 Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(label_names, proba)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)

        st.subheader("💡 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

        st.subheader("📈 Model Evaluation (Test Set)")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

else:
    st.warning("👈 Please upload a CSV file to get started.")
