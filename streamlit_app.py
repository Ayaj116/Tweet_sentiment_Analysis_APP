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

# Initialize ChromaDB Client (Use Persistent Storage to avoid errors)
try:
    client = chromadb.PersistentClient(path="./chroma_db")  # Use persistent storage
    collection = client.get_or_create_collection("covid_tweets")
except Exception as e:
    st.error(f"🚨 Error initializing ChromaDB: {e}")
    st.stop()

# Custom Styling
st.markdown(
    """
    <style>
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
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("🦠 COVID-19 Tweet Sentiment & Similarity Analyzer")
st.write("🔍 Enter a tweet to analyze its sentiment and find similar tweets.")

# Input Section
st.markdown("### 📝 Enter Tweet")
user_input = st.text_area(" ", "", height=150)

# Function to Retrieve Similar Tweets
def retrieve_similar_tweets(query, n_results=3):
    query_embedding = embed_model.encode(query).tolist()
    
    # Ensure the collection exists before querying
    if collection.count() == 0:
        return []
    
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Handle potential missing metadata
    if results and "metadatas" in results and results["metadatas"][0]:
        return [r.get("text", "N/A") for r in results["metadatas"][0]]
    return []

# Analyze Button
if st.button("📊 Analyze & Find Similar Tweets"):
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

        # Retrieve Similar Tweets
        similar_tweets = retrieve_similar_tweets(user_input)

        if similar_tweets:
            st.subheader("🔍 Similar Tweets:")
            for i, tweet in enumerate(similar_tweets, 1):
                st.write(f"**{i}. {tweet}**")
        else:
            st.write("⚠️ No similar tweets found.")
    else:
        st.warning("⚠️ Please enter a tweet for analysis.")

# Sidebar Info
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.write("""
    - 🎯 This app analyzes sentiment & finds similar COVID-19 tweets.
    - 🔥 Uses **DistilBERT** for sentiment classification.
    - ⚡ Powered by **ChromaDB** & **Hugging Face Transformers**.
    """)

    st.markdown("## 🛠 Built With")
    st.write("✅ Python 🐍, Transformers 🤗, Streamlit 🎈, ChromaDB 🔍")

# Footer
st.markdown("---")
st.markdown("🔹 **Created by Ajay** | Powered by AI 🤖")
