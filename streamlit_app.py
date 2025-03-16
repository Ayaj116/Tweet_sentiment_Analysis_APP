import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./fine-tuned-distilbert-covid-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Define label mapping
label_mapping = {
    "LABEL_0": "Positive 😊",  
    "LABEL_1": "Neutral 😐",  
    "LABEL_2": "Negative 😔"  
}

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .stTextInput>div>div>input {
            font-size: 16px !important;
        }
        .stTextArea>div>textarea {
            font-size: 16px !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px 24px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Instructions
st.title("🦠 COVID-19 Tweet Sentiment Analyzer")
st.write("🔍 Enter a tweet below to analyze its sentiment.")

# Input Section
st.markdown("### 📝 Enter Tweet")
user_input = st.text_area(" ", "", height=150)

# Analyze Button
if st.button("📊 Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing sentiment... ⏳"):
            time.sleep(1.5)  # Simulate loading animation

        result = sentiment_pipeline(user_input)
        sentiment_label = result[0]['label']
        sentiment = label_mapping.get(sentiment_label, "Unknown ❓")
        confidence = result[0]['score']

        # Display Sentiment Result
        st.markdown(f"## Predicted Sentiment: **{sentiment}**")

        # Show confidence score
        st.write(f"📈 **Confidence Score:** {confidence:.2f}")
    else:
        st.warning(" Please enter a tweet for analysis. ⚠️")

# Sidebar Info
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.write("""
    - 🎯 This app analyzes the sentiment of COVID-19 related tweets.
    - 🔥 Uses **DistilBERT** model for text classification.
    - 🚀 Developed with **Hugging Face Transformers** & **Streamlit**.
    """)
    
    st.markdown("## 🛠 Built With")
    st.write("✅ Python 🐍, Transformers 🤗, Streamlit 🎈")

# Footer
st.markdown("---")
st.markdown("🔹 **Created by Ajay** | Powered by AI 🤖")
