# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(page_title="Student Predictor", layout="wide")

# Firebase Init
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace("\\\\n", "\n"),
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
        "universe_domain": st.secrets["firebase"]["universe_domain"]
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("xAPI-Edu-Data.csv")

data = load_data()

# Label encoding
label_encoders = {}
for col in ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID',
            'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
            'StudentAbsenceDays']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predict", "📂 History", "📈 Insights", "ℹ️ About"])

# ======================= TAB 1: Predict ========================
with tab1:
    st.header("📊 Predict Student Success")
    st.markdown("Enter details below to predict the student’s academic outcome.")

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox('Gender', label_encoders['gender'].classes_)
            nationality = st.selectbox('Nationality', label_encoders['NationalITy'].classes_)
            place_of_birth = st.selectbox('Place of Birth', label_encoders['PlaceofBirth'].classes_)
            stage_id = st.selectbox('Stage', label_encoders['StageID'].classes_)
            grade_id = st.selectbox('Grade', label_encoders['GradeID'].classes_)
            section_id = st.selectbox('Section', label_encoders['SectionID'].classes_)
            topic = st.selectbox('Topic', label_encoders['Topic'].classes_)
            semester = st.selectbox('Semester', label_encoders['Semester'].classes_)

        with col2:
            raised_hands = st.slider('Times Raised Hands', 0, 100, 10)
            visited_resources = st.slider('Visited Resources', 0, 100, 10)
            announcements_viewed = st.slider('Announcements Viewed', 0, 100, 10)
            discussion = st.slider('Discussion Participation', 0, 100, 10)
            parent_answer_survey = st.selectbox('Parent Answered Survey', label_encoders['ParentAnsweringSurvey'].classes_)
            parent_satisfaction = st.selectbox('Parent School Satisfaction', label_encoders['ParentschoolSatisfaction'].classes_)
            student_absence_days = st.selectbox('Student Absence Days', label_encoders['StudentAbsenceDays'].classes_)
            relation = st.selectbox('Relation', label_encoders['Relation'].classes_)

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        features = {
            'gender': gender,
            'NationalITy': nationality,
            'PlaceofBirth': place_of_birth,
            'StageID': stage_id,
            'GradeID': grade_id,
            'SectionID': section_id,
            'Topic': topic,
            'Semester': semester,
            'raisedhands': raised_hands,
            'VisITedResources': visited_resources,
            'AnnouncementsView': announcements_viewed,
            'Discussion': discussion,
            'ParentAnsweringSurvey': parent_answer_survey,
            'ParentschoolSatisfaction': parent_satisfaction,
            'StudentAbsenceDays': student_absence_days,
            'Relation': relation
        }
        input_df = pd.DataFrame([features])

        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

        input_df = input_df[X_train.columns]
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        class_mapping = {'L': 'At Risk ❌', 'M': 'Likely to Succeed ✅', 'H': 'Excellent 🎓'}
        predicted_label = prediction[0]
        predicted_class = class_mapping.get(predicted_label, 'Unknown')

        st.success(f"🎯 **Predicted Student Status:** {predicted_class}")

        st.subheader("Confidence Levels")
        chart_type = st.radio("Choose chart type:", ["Bar Chart", "Pie Chart"], horizontal=True)
        labels = ['At Risk', 'Likely to Succeed', 'Excellent']
        probs = prediction_proba[0]
        fig, ax = plt.subplots()
        if chart_type == "Bar Chart":
            ax.bar(labels, probs)
            ax.set_ylabel("Probability")
        else:
            ax.pie(probs, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        # Save to Firebase
        log_data = input_df.copy()
        log_data['PredictedLabel'] = predicted_class
        log_data['Proba_AtRisk'] = probs[0]
        log_data['Proba_LikelySuccess'] = probs[1]
        log_data['Proba_Excellent'] = probs[2]
        try:
            db.collection("predictions").add(log_data.to_dict('records')[0])
            st.info("✅ Prediction saved to Firebase")
        except Exception as e:
            st.error(f"⚠️ Could not save to Firebase: {e}")

# ======================= TAB 2: History ========================
with tab2:
    st.header("📂 Prediction History")
    st.markdown("_Coming soon: View past predictions stored in Firebase._")

# ======================= TAB 3: Insights ========================
with tab3:
    st.header("📈 Data Insights")
    st.markdown("_Coming soon: Visualize feature importance, accuracy, and trends._")

# ======================= TAB 4: About ========================
with tab4:
    st.header("ℹ️ About This App")
    st.markdown("""
    This app predicts student academic performance using machine learning. 
    Built by **Thushar Akumarthi** using Streamlit, Firebase, and Random Forest.
    
    Future features:
    - SHAP explanations
    - Dataset comparison
    - PDF report generation
    - User login
    """)
