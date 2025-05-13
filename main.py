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

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
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

# Title
st.title("Student Success Prediction Dashboard")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("xAPI-Edu-Data.csv")
    return data

data = load_data()

# Label encoders
label_encoders = {}
for col in ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID',
            'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
            'StudentAbsenceDays']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features & target
X = data.drop('Class', axis=1)
y = data['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Sidebar form
st.sidebar.header("Input Student Details")
def user_input_features():
    gender = st.sidebar.selectbox('Gender', label_encoders['gender'].classes_)
    nationality = st.sidebar.selectbox('Nationality', label_encoders['NationalITy'].classes_)
    place_of_birth = st.sidebar.selectbox('Place of Birth', label_encoders['PlaceofBirth'].classes_)
    stage_id = st.sidebar.selectbox('Stage', label_encoders['StageID'].classes_)
    grade_id = st.sidebar.selectbox('Grade', label_encoders['GradeID'].classes_)
    section_id = st.sidebar.selectbox('Section', label_encoders['SectionID'].classes_)
    topic = st.sidebar.selectbox('Topic', label_encoders['Topic'].classes_)
    semester = st.sidebar.selectbox('Semester', label_encoders['Semester'].classes_)
    raised_hands = st.sidebar.slider('Times Raised Hands', 0, 100, 10)
    visited_resources = st.sidebar.slider('Visited Resources', 0, 100, 10)
    announcements_viewed = st.sidebar.slider('Announcements Viewed', 0, 100, 10)
    discussion = st.sidebar.slider('Discussion Participation', 0, 100, 10)
    parent_answer_survey = st.sidebar.selectbox('Parent Answered Survey', label_encoders['ParentAnsweringSurvey'].classes_)
    parent_satisfaction = st.sidebar.selectbox('Parent School Satisfaction', label_encoders['ParentschoolSatisfaction'].classes_)
    student_absence_days = st.sidebar.selectbox('Student Absence Days', label_encoders['StudentAbsenceDays'].classes_)
    relation = st.sidebar.selectbox('Relation', label_encoders['Relation'].classes_)

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
    return pd.DataFrame([features])

input_df = user_input_features()

# Encode inputs
for col in label_encoders:
    val = input_df[col].values[0]
    if val not in label_encoders[col].classes_:
        st.error(f"Invalid input: '{val}' not found in training data for column '{col}'.")
        st.stop()
    input_df[col] = label_encoders[col].transform(input_df[col])

# Prediction
input_df = input_df[X_train.columns]
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader('Prediction')
class_mapping = {'L': 'At Risk', 'M': 'Likely to Succeed', 'H': 'Excellent'}
predicted_label = prediction[0]
predicted_class = class_mapping.get(predicted_label, 'Unknown')
st.write(f"Predicted Student Status: **{predicted_class}**")

# Probabilities
st.subheader('Prediction Probabilities')
st.write(prediction_proba)

# Chart
chart_type = st.sidebar.radio("Choose chart type:", ["Bar Chart", "Pie Chart"])
labels = ['At Risk', 'Likely to Succeed', 'Excellent']
probs = prediction_proba[0]
fig, ax = plt.subplots()
if chart_type == "Bar Chart":
    ax.bar(labels, probs, color=['red', 'green', 'blue'])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
else:
    ax.pie(probs, labels=labels, autopct='%1.1f%%', colors=['red', 'green', 'blue'])
    ax.set_title("Prediction Probabilities")
st.pyplot(fig)

# Save to Firebase Firestore
log_data = input_df.copy()
log_data['PredictedLabel'] = predicted_class
log_data['Proba_AtRisk'] = probs[0]
log_data['Proba_LikelySuccess'] = probs[1]
log_data['Proba_Excellent'] = probs[2]
log_dict = log_data.to_dict('records')[0]

try:
    db.collection("predictions").add(log_dict)
    st.success("Prediction saved to Firebase Firestore ✅")
except Exception as e:
    st.error(f"Failed to save to Firebase: {e}")
