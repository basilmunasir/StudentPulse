import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from chatbot import get_study_advice

# Page config
st.set_page_config(
    page_title="StudentPulse",
    page_icon="🎓",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/studentpulse_model.keras')

@st.cache_data
def load_data():
    df = pd.read_csv('data/student_habits_performance.csv')
    df['parental_education_level'] = df['parental_education_level'].fillna(
        df['parental_education_level'].mode()[0]
    )
    return df

model = load_model()
df = load_data()

# Scale setup
features = [
    'study_hours_per_day',
    'mental_health_rating',
    'exercise_frequency',
    'sleep_hours',
    'social_media_hours',
    'netflix_hours'
]
scaler = StandardScaler()
scaler.fit(df[features])

# Header
st.title("🎓 StudentPulse")
st.subheader("AI-Powered Student Life Intelligence Dashboard")
st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Your Habits")
    
    # Track remaining hours
    total_available = 24.0
    
    study_hours = st.slider("Study Hours Per Day", 0.0, total_available, 4.0, 0.5)
    remaining_after_study = total_available - study_hours
    
    sleep_hours = st.slider("Sleep Hours", 0.0, remaining_after_study, 
                            min(7.0, remaining_after_study), 0.5)
    remaining_after_sleep = remaining_after_study - sleep_hours
    
    social_media_hours = st.slider("Social Media Hours", 0.0, remaining_after_sleep, 
                                   min(2.0, remaining_after_sleep), 0.5)
    remaining_after_social = remaining_after_sleep - social_media_hours
    
    netflix_hours = st.slider("Netflix Hours", 0.0, remaining_after_social, 
                              min(1.0, remaining_after_social), 0.5)
    remaining_after_netflix = remaining_after_social - netflix_hours
    
    exercise_frequency = st.slider("Exercise Days Per Week", 0, 7, 3)
    mental_health_rating = st.slider("Mental Health Rating (1-10)", 1, 10, 7)
    
    # Show remaining hours
    st.info(f"⏰ Remaining hours in your day: {remaining_after_netflix:.1f} hrs")

with col2:
    st.header("🔮 Your Prediction")
    
    if st.button("Predict My Score!", use_container_width=True):
        # Prepare input
        input_data = np.array([[
            study_hours,
            mental_health_rating,
            exercise_frequency,
            sleep_hours,
            social_media_hours,
            netflix_hours
        ]])
        
        input_scaled = scaler.transform(input_data)
        predicted_score = model.predict(input_scaled)[0][0]
        predicted_score = float(np.clip(predicted_score, 0, 100))
        
        # Show score
        st.metric("Predicted Exam Score", f"{predicted_score:.1f} / 100")
        
        # Score feedback
        if predicted_score >= 80:
            st.success("Excellent! You're on track for great results! 🌟")
        elif predicted_score >= 60:
            st.warning("Good! A few improvements can push you higher! 💪")
        else:
            st.error("Let's work on your habits to improve your score! 📚")
        
        # Get AI advice
        st.header("🤖 AI Study Advisor")
        with st.spinner("Getting personalized advice..."):
            student_data = {
                'study_hours': study_hours,
                'sleep_hours': sleep_hours,
                'social_media_hours': social_media_hours,
                'netflix_hours': netflix_hours,
                'exercise_frequency': exercise_frequency,
                'mental_health_rating': mental_health_rating,
                'predicted_score': round(predicted_score, 1)
            }
            advice = get_study_advice(student_data)
            st.write(advice)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using TensorFlow, Streamlit and Groq AI")