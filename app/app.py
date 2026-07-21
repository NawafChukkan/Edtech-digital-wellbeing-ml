
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Digital Wellbeing Predictor", page_icon="🧠", layout="centered")

st.title("🧠 EdTech & Digital Wellbeing Predictor")
st.markdown("An AI-driven decision support system to identify student burnout risk based on digital usage patterns.")

# 2. Sidebar for Live Demo Inputs
st.sidebar.header("📊 Student Digital Telemetry")
screen_hours = st.sidebar.slider("Daily Screen Time (Hours)", 0.0, 14.0, 4.5)
notifications = st.sidebar.slider("Daily Notifications Count", 5, 400, 85)
sleep_hours = st.sidebar.slider("Average Sleep (Hours)", 3.0, 10.0, 7.0)
anxiety = st.sidebar.slider("Anxiety Score (0-27 Scale)", 0, 27, 12)

# 3. Model Training (Cached for speed during presentation)
@st.cache_resource
def load_and_train_model():
    # Replace with your actual GitHub username
    DATA_URL = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/edtech-digital-wellbeing-ml/main/social_media_screentime_mental_health_2026.csv"
    
    try:
        df = pd.read_csv(DATA_URL)
    except:
        # Fallback for local testing
        df = pd.read_csv("social_media_screentime_mental_health_2026.csv")
        
    df['avg_sleep_hours'].fillna(df['avg_sleep_hours'].median(), inplace=True)
    
    # Train lightweight model on core features
    X = df[['daily_screen_hours', 'daily_notifications', 'avg_sleep_hours', 'anxiety_score_0to27']]
    y = df['wellbeing_band']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model

model = load_and_train_model()

# 4. Live Prediction Logic
input_df = pd.DataFrame([[screen_hours, notifications, sleep_hours, anxiety]], 
                         columns=['daily_screen_hours', 'daily_notifications', 'avg_sleep_hours', 'anxiety_score_0to27'])

prediction = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0]

# 5. Display Results
st.divider()
st.subheader("Predictive Risk Assessment")

if prediction == "At-risk":
    st.error(f"⚠️ **High Risk Level Detected (`{prediction}`)**")
    st.caption("Intervention Recommendation: Automated screen-time throttling and proactive referral to university mental health services.")
elif prediction == "Moderate":
    st.warning(f"🟡 **Moderate Risk Level (`{prediction}`)**")
    st.caption("Intervention Recommendation: Push notifications suggesting digital detox breaks and sleep hygiene tips.")
else:
    st.success(f"✅ **Optimal Wellbeing (`{prediction}`)**")
    st.caption("User exhibits balanced digital habits.")

# Display Probabilities
st.markdown("### Model Confidence Metrics")
prob_df = pd.DataFrame({'Category': model.classes_, 'Probability': [f"{p*100:.1f}%" for p in probs]})
st.table(prob_df)
