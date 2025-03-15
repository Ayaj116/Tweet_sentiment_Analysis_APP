import streamlit as st
import torch
st.title("🎈")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./fine-tuned-distilbert-covid-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Define label mapping (Update based on your dataset)
label_mapping = {
    "LABEL_0": "pos",   # postive sentiment
    "LABEL_1": "neutral",  # Neutral sentiment
    "LABEL_2": "neg"   # negtive sentiment
}

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Streamlit UI
st.title("COVID-19 Tweet Sentiment Analyzer")
st.write("Enter a tweet to analyze its sentiment.")

user_input = st.text_area("Enter Tweet:", "")

if st.button("Analyze"):
    if user_input:
        result = sentiment_pipeline(user_input)
        sentiment_label = result[0]['label']
        sentiment = label_mapping.get(sentiment_label, "unknown")  # Map labels
        confidence = result[0]['score']

        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.2f}")
    else:
        st.warning("Please enter a tweet for analysis.")

