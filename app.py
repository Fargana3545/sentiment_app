import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pickle
import faiss
import numpy as np

# Load sentiment model
model_path = "models/sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load FAISS index
faiss_index = faiss.read_index("faiss_reviews.index")

# Load review chunks
with open("review_embeddings.pkl", "rb") as f:
    review_chunks = pickle.load(f)

# Load same embedding model
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("📊 Sentiment + Similar Review Finder")
st.write("Enter a product review to analyze its sentiment and fetch similar reviews.")

user_input = st.text_input("✍️ Enter a review:")

if user_input:
    with st.spinner("Analyzing..."):
        # Sentiment prediction
        result = sentiment_pipeline(user_input)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)

        st.success(f"Sentiment: **{label}** ({score}%)")

        # Encode query for similarity search
        query_vec = embed_model.encode(user_input).astype("float32").reshape(1, -1)
        distances, indices = faiss_index.search(query_vec, k=5)

        # Display similar reviews
        st.subheader("🔍 Top 5 Similar Reviews:")
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(review_chunks):
                st.markdown(f"**{i+1}.** {review_chunks[idx]}")
            else:
                st.warning(f"Index {idx} out of range.")
