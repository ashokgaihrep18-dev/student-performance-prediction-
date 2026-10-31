import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("Student Exam Score Predictor")
st.write("Enter the student's details and click Predict.")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("Hours studied per week", 1, 44, 20)
    attendance = st.slider("Attendance (%)", 60, 100, 80)
    sleep = st.slider("Sleep hours per night", 4, 10, 7)
    previous = st.slider("Previous score", 50, 100, 75)
    tutoring = st.slider("Tutoring sessions per month", 0, 8, 1)
    physical = st.slider("Physical activity (hours/week)", 0, 6, 3)
    gender = st.selectbox("Gender", ["Male", "Female"])
    school = st.selectbox("School type", ["Public", "Private"])
    distance = st.selectbox("Distance from home", ["Near", "Moderate", "Far"])

with col2:
    parental_inv = st.selectbox("Parental involvement", ["Low", "Medium", "High"])
    resources = st.selectbox("Access to resources", ["Low", "Medium", "High"])
    motivation = st.selectbox("Motivation level", ["Low", "Medium", "High"])
    income = st.selectbox("Family income", ["Low", "Medium", "High"])
    teacher = st.selectbox("Teacher quality", ["Low", "Medium", "High"])
    peer = st.selectbox("Peer influence", ["Positive", "Neutral", "Negative"])
    parent_edu = st.selectbox("Parental education", ["High School", "College", "Postgraduate"])
    extra = st.selectbox("Extracurricular activities", ["Yes", "No"])
    internet = st.selectbox("Internet access", ["Yes", "No"])
    disability = st.selectbox("Learning disabilities", ["No", "Yes"])

if st.button("Predict"):
    raw = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Sleep_Hours": sleep,
        "Previous_Scores": previous,
        "Tutoring_Sessions": tutoring,
        "Physical_Activity": physical,
        "Gender": gender,
        "School_Type": school,
        "Distance_from_Home": distance,
        "Parental_Involvement": parental_inv,
        "Access_to_Resources": resources,
        "Motivation_Level": motivation,
        "Family_Income": income,
        "Teacher_Quality": teacher,
        "Peer_Influence": peer,
        "Parental_Education_Level": parent_edu,
        "Extracurricular_Activities": extra,
        "Internet_Access": internet,
        "Learning_Disabilities": disability,
    }])

    # Same encoding as training, then force the exact same columns and order
    encoded = pd.get_dummies(raw, dtype=int)
    encoded = encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(encoded)[0]
    prediction = max(0, min(100, prediction))
    st.success(f"Predicted exam score: {prediction:.1f}")