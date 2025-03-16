import streamlit as st
import time
import sys

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from sentence_transformers import SentenceTransformer
    import chromadb
except ModuleNotFoundError as e:
    st.error(f"⚠️ Missing dependency: {e}. Install required packages before running.")

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # If pysqlite3 is not available, fall back to the system SQLite

# Load Sentiment Analysis Model
MODEL_PATH = "./fine-tuned-distilbert-covid-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define label mapping
label_mapping = {
    "LABEL_0": "Positive 😊",
    "LABEL_1": "Neutral 😐",
    "LABEL_2": "Negative 😔"
}

# Load Sentence Transformer Model for Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path="./chroma_db")  # Ensure persistence
collection = client.get_or_create_collection("covid_tweets")

# Custom Styling
st.markdown("""
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
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🦠 COVID-19 Tweet Sentiment & Similarity Analyzer")
st.write("🔍 Enter a tweet to analyze its sentiment and find similar tweets.")

# Input Section
st.markdown("### 📝 Enter Tweet")
user_input = st.text_area(" ", "", height=150)

# Function to Retrieve Similar Tweets
def retrieve_similar_tweets(query, n_results=3):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    if results and results.get("metadatas"):
        return [(r.get("text", "N/A"), r.get("label", "Unknown")) for r in results["metadatas"][0]]
    
    return []

# Analyze Button
if st.button("📊 Analyze & Find Similar Tweets"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment... ⏳"):
            time.sleep(1.5)  # Simulate loading animation

        # Sentiment Analysis
        result = sentiment_pipeline(user_input)
        sentiment_label = result[0]['label']
        sentiment = label_mapping.get(sentiment_label, "Unknown ❓")
        confidence = result[0]['score']

        
