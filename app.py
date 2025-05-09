import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import torch
import pickle
import faiss
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === Load Sentiment Analysis Model from Hugging Face ===
model_path = "Fargana/sentiment-model-v1"  # ✅ FIXED

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# === Load FAISS Index and Review Chunks ===
faiss_index = faiss.read_index("faiss_reviews.index")
with open("review_embeddings.pkl", "rb") as f:
    review_chunks = pickle.load(f)

# === Load Embedding Model ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Sentiment Label Map ===
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
        result = sentiment_pipeline(user_input)[0]
        label = result["label"]
        readable_label = label_map.get(label, label)
        score = round(result["score"] * 100, 2)

        st.markdown("## 🔎 Sentiment Result")
        st.success(f"**Sentiment:** {readable_label} ({score}%)")

        # --- Bar Chart ---
        st.markdown("#### 📊 Sentiment Confidence")
        fig_bar, ax_bar = plt.subplots()
        color = 'green' if label == "LABEL_2" else 'orange' if label == "LABEL_1" else 'red'
        ax_bar.bar([readable_label], [score], color=color)
        ax_bar.set_ylim([0, 100])
        ax_bar.set_ylabel("Confidence (%)")
        st.pyplot(fig_bar)

        # --- Confidence Level ---
        if score > 80:
            st.info("✅ The model is very confident.")
        elif score > 60:
            st.info("ℹ️ The model is moderately confident.")
        else:
            st.warning("⚠️ The model is not very confident. Use with caution.")

        # --- Similar Review Search ---
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
    except ValueError as e:
        st.warning("⚠️ Not enough text to generate a word cloud. Please check your review data.")

# === Clear Input Button ===
if st.button("🧹 Clear Input"):
    st.experimental_rerun()

# === Footer ===
st.markdown("---")
st.caption("Developed with ❤️ for insightful review analysis.")


