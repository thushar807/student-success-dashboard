import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 🎨 Page configuration
st.set_page_config(page_title="🎓 Student Success Predictor", layout="wide")
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1522071820081-009f0129c71c');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🎓 Student Success Predictor Dashboard</h1>", unsafe_allow_html=True)

# 🌟 Smart Pass/Fail inference
def infer_pass_fail(df):
    proxy_cols = []
    score = pd.Series(0, index=df.index, dtype=float)

    if 'StudyTimeWeekly' in df.columns:
        score += df['StudyTimeWeekly'].fillna(0)
        proxy_cols.append('StudyTimeWeekly')

    if 'Absences' in df.columns:
        score -= df['Absences'].fillna(0)
        proxy_cols.append('Absences')

    if 'Tutoring' in df.columns:
        score += df['Tutoring'].fillna(0) * 5
        proxy_cols.append('Tutoring')

    if 'AnnouncementsView' in df.columns:
        score += df['AnnouncementsView'].fillna(0)
        proxy_cols.append('AnnouncementsView')

    if score.std() != 0:
        score = (score - score.mean()) / score.std()

    inferred = (score > score.mean()).astype(int)
    df['Pass_Fail'] = inferred.map({1: 'Pass', 0: 'Fail'})
    return df, proxy_cols

# 📂 Upload section
with st.sidebar:
    st.markdown("### 📁 Upload Student Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # 🔍 Step 1: Infer PassStatus
    if "PassStatus" not in df.columns:
        grade_cols = ["G3", "final_grade", "FinalGrade", "grade"]
        found = False
        for col in grade_cols:
            if col in df.columns:
                df["PassStatus"] = df[col].apply(lambda x: "Pass" if x >= 10 else "Fail")
                found = True
                break

        if not found:
            df, inferred_cols = infer_pass_fail(df)
            if "Pass_Fail" in df.columns:
                df["PassStatus"] = df["Pass_Fail"]
                df.drop(columns=["Pass_Fail"], inplace=True)
                st.info(f"💡 Pass/Fail status inferred from: {', '.join(inferred_cols)}")
            else:
                st.error("❌ Could not determine pass/fail. Please use a dataset with grades.")
                st.stop()

    # ⚙️ Step 2: Model setup
    st.subheader("⚙️ Model Settings")
    target_column = "PassStatus"
    feature_columns = st.multiselect("🧠 Select input features for prediction:", options=[col for col in df.columns if col != target_column])

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

        # 📝 Step 3: Prediction form
        st.subheader("📝 Enter Student Details")
        with st.container():
            st.markdown("<div style='background-color:#f9f9f9; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
            user_input = {}
            for col in feature_columns:
                if col in encoders:
                    user_input[col] = st.selectbox(f"{col}", encoders[col].classes_)
                else:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    user_input[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=min_val)
            st.markdown("</div>", unsafe_allow_html=True)

        input_df = pd.DataFrame([user_input])
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        label_names = encoders[target_column].classes_ if target_column in encoders else sorted(y.unique())

        # 🎯 Step 4: Output
        st.subheader("🎯 Prediction Result")
        st.success(f"**Predicted Class:** {encoders[target_column].inverse_transform([prediction])[0] if target_column in encoders else prediction}")

        st.subheader("📊 Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(label_names, proba, color='#1f77b4')
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
        st.dataframe(pd.DataFrame(report).transpose())
else:
    st.warning("👈 Please upload a CSV file to begin.")
