import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import pickle
import faiss
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === Load Sentiment Analysis Model ===
# Replace this with your original sentiment model and logic
# For example, if you're using a PyTorch model, load it here and implement the logic

# Example: Sentiment model loading (add your own)
# sentiment_model = torch.load("your_model.pth")
# sentiment_model.eval()

# === Load FAISS Index and Review Chunks ===
try:
    faiss_index = faiss.read_index("faiss_reviews.index")
    with open("review_embeddings.pkl", "rb") as f:
        review_chunks = pickle.load(f)
    print("FAISS index and review chunks loaded successfully!")
except Exception as e:
    st.error("Error loading FAISS index or review embeddings. Please ensure the files are present.")
    faiss_index = None
    review_chunks = []

# === Load Embedding Model ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Sentiment Label Map (Replace with your own model's labels) ===
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# === Streamlit UI ===
st.set_page_config(page_title="Sentiment App")
st.title("📊 Sentiment Analysis + Similar Review Finder")

st.markdown("### ✍️ Enter a review:")
user_input = st.text_area("Your Review", height=150)

if user_input:
    with st.spinner("Analyzing..."):
        # --- Sentiment Analysis ---
        # Your sentiment analysis logic here
        # For example, if you are using a model like distilBERT or a PyTorch model:
        # inputs = sentiment_model(user_input)
        # probabilities = torch.softmax(outputs.logits, dim=1)
        # confidence = float(torch.max(probabilities).item()) * 100
        
        # For now, let's assume a sample sentiment result
        result = {"label": "LABEL_2", "score": 0.85}  # Replace with actual output from your sentiment model
        label = result["label"]
        readable_label = label_map.get(label, label)
        score = round(result["score"] * 100, 2)

        st.markdown("## 🔎 Sentiment Result")
        st.success(f"**Sentiment:** {readable_label} ({score}%)")

        # --- Sentiment Confidence ---
        confidence = score  # Adjust if needed from your model's confidence score
        if confidence > 80:
            st.info("✅ The model is very confident.")
        elif confidence > 60:
            st.info("ℹ️ The model is moderately confident.")
        else:
            st.warning("⚠️ The model is not very confident. Use with caution.")

        # --- Similar Review Search ---
        if faiss_index and review_chunks:
            st.markdown("## 🔍 Top 5 Similar Reviews")
            query_vec = embed_model.encode(user_input).astype("float32").reshape(1, -1)
            distances, indices = faiss_index.search(query_vec, k=10)

            shown = 0
            for idx in indices[0]:
                if 0 <= idx < len(review_chunks):
                    retrieved = review_chunks[idx].strip()
                    if retrieved != user_input.strip():
                        shown += 1
                        st.markdown(f"**{shown}.** {retrieved}")
                        if shown == 5:
                            break

            if shown == 0:
                st.info("No similar reviews found that are different from your input.")

# === Word Cloud Section ===
if review_chunks:
    st.markdown("## 🎨 Word Cloud of Reviews")
    all_text = " ".join(review_chunks)

    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    except ValueError:
        st.warning("⚠️ Not enough text to generate a word cloud. Please check your review data.")

# === Clear Input Button ===
if st.button("🧹 Clear Input"):
    st.experimental_rerun()

# === Footer ===
st.markdown("---")
st.caption("Developed with ❤️ for insightful review analysis.")

