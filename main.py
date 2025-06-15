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

# Page setup
st.set_page_config(page_title="Student Predictor", layout="wide")

# Firebase Initialization with safe fallback
try:
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
    firebase_ready = True
except Exception as e:
    st.warning(f"⚠️ Firebase not connected: {e}")
    firebase_ready = False

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("xAPI-Edu-Data.csv")

data = load_data()

# Encode labels
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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predict", "📂 History", "📈 Insights", "ℹ️ About"])

# Tab 1: Predict
with tab1:
    st.header("📊 Predict Student Success")
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

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        class_map = {'L': 'At Risk ❌', 'M': 'Likely to Succeed ✅', 'H': 'Excellent 🎓'}
        st.success(f"🎯 **Predicted Student Status:** {class_map.get(prediction)}")

        st.subheader("Confidence Breakdown")
        chart = st.radio("Choose chart type", ["Bar Chart", "Pie Chart"], horizontal=True)
        labels = ['At Risk', 'Likely to Succeed', 'Excellent']
        fig, ax = plt.subplots()
        if chart == "Bar Chart":
            ax.bar(labels, prediction_proba)
            ax.set_ylabel("Probability")
        else:
            ax.pie(prediction_proba, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        if firebase_ready:
            log = input_df.copy()
            log['PredictedLabel'] = class_map.get(prediction)
            log['Proba_AtRisk'] = prediction_proba[0]
            log['Proba_LikelySuccess'] = prediction_proba[1]
            log['Proba_Excellent'] = prediction_proba[2]
            try:
                db.collection("predictions").add(log.to_dict('records')[0])
                st.info("✅ Data saved to Firebase")
            except Exception as e:
                st.error(f"🔥 Firebase save failed: {e}")

# Tab 2: History
with tab2:
    st.header("📂 History (Coming Soon)")

# Tab 3: Insights
with tab3:
    st.header("📈 Insights (Coming Soon)")

# Tab 4: About
with tab4:
    st.header("ℹ️ About This App")
    st.markdown("Built by **Thushar Akumarthi** using Streamlit + Firebase + ML.\n\nFuture: SHAP plots, CSV uploads, PDF reports.")
