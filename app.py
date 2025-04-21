import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page configuration
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with example messages
st.sidebar.header("Example Messages")
st.sidebar.markdown("Click on any example to use it:")

# Example messages in the sidebar with buttons
if st.sidebar.button("📨 SPAM: Urgent Gift Card", help="Click to use this example"):
    st.session_state.user_input = "URGENT: You've WON a FREE $1,000 WALMART gift voucher! Text 'WIN' to 80580 to claim now. Msg&data rates may apply."

if st.sidebar.button("📨 SPAM: Free Prize Entry", help="Click to use this example"):
    st.session_state.user_input = "FREE ENTRY to WIN a FREE AUDI TT! Text AUDI to 81122! Txt STOP to cancel, std msg rate applies."

if st.sidebar.button("📨 SPAM: Account Alert", help="Click to use this example"):
    st.session_state.user_input = "IMPORTANT: Your account has been suspended. Call our security team at 02089999999 immediately to reactivate your account."

if st.sidebar.button("✉️ HAM: Meeting Request", help="Click to use this example"):
    st.session_state.user_input = "Hey, can we reschedule our meeting to 3pm tomorrow? I've got a doctor's appointment in the morning. Thanks!"

if st.sidebar.button("✉️ HAM: Grocery Reminder", help="Click to use this example"):
    st.session_state.user_input = "Don't forget to pick up milk and eggs on your way home. Also, what time will you be back for dinner?"

# Add explanation of the classifier in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown("""
This classifier uses machine learning to identify whether a text message is spam or not (ham).
It analyzes the text patterns and compares them with known spam and ham messages it was trained on.

""")

# Initialize session state for the text input if it doesn't exist
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# App title with custom styling
st.markdown('<h1 class="main-header">📱 SMS Spam Classifier</h1>', unsafe_allow_html=True)

# Main content area
st.markdown("### Enter your message below")

# Text area with session state value
user_input = st.text_area(
    "",
    value=st.session_state.user_input,
    height=150,
    placeholder="Type or paste your SMS message here, or select an example from the sidebar..."
)

# Update session state when user types
if user_input != st.session_state.user_input:
    st.session_state.user_input = user_input

# Horizontal layout for buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    classify_button = st.button("Classify Message", type="primary", use_container_width=True)

# Process classification
if classify_button:
    if user_input:
        try:
            # Load model and vectorizer
            model = joblib.load('spam_classifier_model.joblib')
            vectorizer = joblib.load('tfidf_vectorizer.joblib')
            
            # Transform input and predict
            input_transformed = vectorizer.transform([user_input])
            prediction = model.predict(input_transformed)
            probability = model.predict_proba(input_transformed)
            
            # Display result with styling
            if prediction[0] == 'spam':
                spam_prob = probability[0][1] if probability[0].size > 1 else probability[0][0]
                st.markdown(
                    f'<div class="result-box" style="background-color: #FFCDD2;">🚨 SPAM DETECTED<br>'
                    f'Confidence: {spam_prob:.2%}</div>',
                    unsafe_allow_html=True
                )
            else:
                ham_prob = probability[0][0] if prediction[0] == 'ham' else probability[0][1]
                st.markdown(
                    f'<div class="result-box" style="background-color: #C8E6C9;">✅ SAFE MESSAGE (HAM)<br>'
                    f'Confidence: {ham_prob:.2%}</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"An error occurred: {e}. Make sure model files exist.")
    else:
        st.warning("Please enter a message to classify.")

# Usage statistics in a dashboard-like format
if user_input:
    st.markdown('<h3 class="sub-header">Message Analysis</h3>', unsafe_allow_html=True)
    
    # Calculate simple text statistics
    char_count = len(user_input)
    word_count = len(user_input.split())
    has_url = "http" in user_input.lower() or "www" in user_input.lower()
    has_numbers = any(char.isdigit() for char in user_input)
    uppercase_ratio = sum(1 for c in user_input if c.isupper()) / max(1, len(user_input))
    exclamation_count = user_input.count('!')
    
    # Display stats in columns
    st_col1, st_col2, st_col3, st_col4, st_col5 = st.columns(5)
    
    st_col1.metric(label="Characters", value=char_count)
    st_col2.metric(label="Words", value=word_count)
    st_col3.metric(label="URLs", value="Yes" if has_url else "No", delta="Risk Factor" if has_url else None)
    st_col4.metric(label="Numbers", value="Yes" if has_numbers else "No")
    st_col5.metric(label="UPPERCASE %", value=f"{uppercase_ratio:.0%}", delta="High" if uppercase_ratio > 0.3 else None)

# Footer
st.markdown('<div class="footer">Made by Dipesh Sharma</div>', unsafe_allow_html=True)