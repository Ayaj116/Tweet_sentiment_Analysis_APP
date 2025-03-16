import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from chromadb import Client
from sentence_transformers import SentenceTransformer

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


# Use in-memory mode (no persistence)
client = Client()
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

    if results and "metadatas" in results and results["metadatas"]:
        return [(r["text"], r.get("label", "Unknown")) for r in results["metadatas"][0]]
    return []

# Analyze Button
if st.button("📊 Analyze & Find Similar Tweets"):
    if user_input:
        with st.spinner("Analyzing sentiment... ⏳"):
            time.sleep(1.5)  # Simulate loading animation

        # Sentiment Analysis
        result = sentiment_pipeline(user_input)
        sentiment_label = result[0]['label']
        sentiment = label_mapping.get(sentiment_label, "Unknown ❓")
        confidence = result[0]['score']

        # Display Sentiment Result
        st.markdown(f"## Predicted Sentiment: **{sentiment}**")
        st.write(f"📈 **Confidence Score:** {confidence:.2f}")

        # Retrieve Similar Tweets
        similar_tweets = retrieve_similar_tweets(user_input)

        if similar_tweets:
            st.subheader("🔍 Similar Tweets:")
            for i, (tweet, label) in enumerate(similar_tweets, 1):
                st.write(f"**{i}. {tweet}** (Label: {label})")
        else:
            st.write("⚠️ No similar tweets found.")
    else:
        st.warning(" Please enter a tweet for analysis. ⚠️")

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
