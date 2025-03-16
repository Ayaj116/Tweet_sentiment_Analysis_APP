import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # If pysqlite3 is not available, fall back to the system SQLite

import chromadb
from sentence_transformers import SentenceTransformer

# Load Sentiment Analysis Model
MODEL_PATH = "./fine-tuned-distilbert-covid-sentiment"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"🚨 Error loading sentiment model: {e}")
    st.stop()

# Define label mapping
label_mapping = {
    "LABEL_0": "Positive 😊",
    "LABEL_1": "Neutral 😐",
    "LABEL_2": "Negative 😔"
}

# Load Sentence Transformer Model for Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB Client (Use Persistent Storage)
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("covid_tweets")
except Exception as e:
    st.error(f"🚨 Error initializing ChromaDB: {e}")
    st.stop()

# Streamlit UI
st.title("🦠 COVID-19 Tweet Sentiment & Similarity Analyzer")
st.write("🔍 Enter a tweet to analyze its sentiment and improve similar tweet labels.")

# Input Section
st.markdown("### 📝 Enter Tweet")
user_input = st.text_area(" ", "", height=150)

# Function to Retrieve & Re-analyze Similar Tweets
def retrieve_and_update_tweets(query, n_results=3):
    query_embedding = embed_model.encode(query).tolist()
    
    # Ensure the collection exists before querying
    if collection.count() == 0:
        return []
    
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Process retrieved tweets
    updated_tweets = []
    if results and "metadatas" in results and results["metadatas"][0]:
        for r in results["metadatas"][0]:
            tweet_text = r.get("text", "N/A")

            # Re-run sentiment analysis
            sentiment_result = sentiment_pipeline(tweet_text)
            new_label = sentiment_result[0]["label"]
            confidence = sentiment_result[0]["score"]

            updated_tweets.append((tweet_text, label_mapping.get(new_label, "Unknown ❓"), confidence))

    return updated_tweets

# Analyze Button
if st.button("📊 Analyze & Improve Sentiment Labels"):
    if user_input:
        with st.spinner("Analyzing sentiment... ⏳"):
            time.sleep(1.5)  # Simulate loading animation
        
        # Sentiment Analysis
        try:
            result = sentiment_pipeline(user_input)
            sentiment_label = result[0]["label"]
            sentiment = label_mapping.get(sentiment_label, "Unknown ❓")
            confidence = result[0]["score"]
        except Exception as e:
            st.error(f"🚨 Error analyzing sentiment: {e}")
            st.stop()

        # Display Sentiment Result
        st.markdown(f"## Predicted Sentiment: **{sentiment}**")
        st.write(f"📈 **Confidence Score:** {confidence:.2f}")

        # Retrieve & Update Similar Tweets
        updated_tweets = retrieve_and_update_tweets(user_input)

        if updated_tweets:
            st.subheader("🔍 Similar Tweets (Updated Sentiment Labels):")
            for i, (tweet, label, conf) in enumerate(updated_tweets, 1):
                st.write(f"**{i}. {tweet}**")
                st.markdown(f"🟢 **Corrected Sentiment:** {label} | 🔍 **Confidence:** {conf:.2f}")
        else:
            st.write("⚠️ No similar tweets found.")
    else:
        st.warning("⚠️ Please enter a tweet for analysis.")

# Sidebar Info
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.write("""
    - 🎯 This app analyzes sentiment & **improves similar tweet labels**.
    - 🔥 Uses **DistilBERT** for sentiment classification.
    - ⚡ Powered by **ChromaDB** & **Hugging Face Transformers**.
    """)

    st.markdown("## 🛠 Built With")
