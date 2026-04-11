from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_study_advice(student_data: dict) -> str:
    """
    Takes student data and returns personalized study advice
    """
    
    prompt = f"""
    You are StudentPulse AI — a friendly and motivating student advisor.
    
    A student has the following habits and performance data:
    - Study Hours Per Day: {student_data['study_hours']}
    - Sleep Hours: {student_data['sleep_hours']}
    - Social Media Hours: {student_data['social_media_hours']}
    - Netflix Hours: {student_data['netflix_hours']}
    - Exercise Frequency (per week): {student_data['exercise_frequency']}
    - Mental Health Rating (1-10): {student_data['mental_health_rating']}
    - Predicted Exam Score: {student_data['predicted_score']}
    
    Give them:
    1. A short encouraging message based on their habits
    2. Top 3 specific improvements they can make
    3. A motivational closing line
    
    Keep it friendly, short and practical. Talk directly to the student.
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    return chat_completion.choices[0].message.content


# Test it
if __name__ == "__main__":
    test_student = {
        'study_hours': 4,
        'sleep_hours': 6,
        'social_media_hours': 3,
        'netflix_hours': 2,
        'exercise_frequency': 2,
        'mental_health_rating': 7,
        'predicted_score': 72
    }
    
    advice = get_study_advice(test_student)
    print(advice)