import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from textblob import TextBlob
import pandas as pd

st.set_page_config(page_title="News Article Classifier", layout="wide")

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_path="./news_classifier_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

@st.cache_data(show_spinner=True)
def load_categories(dataset_path="newscatcher_dataset.csv"):
    df = pd.read_csv(dataset_path)
    categories = df['category'].unique().tolist()
    return categories

# Load model and categories
tokenizer, model = load_model_and_tokenizer()
categories = load_categories()

st.title("📰 News Article Classifier (BERT + NewsCatcher)")

article = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if not article.strip():
        st.warning("Please enter a news article.")
    else:
        # Tokenize and predict
        inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1).squeeze().tolist()
            pred_index = int(torch.argmax(outputs.logits))
        
        predicted_category = categories[pred_index]
        confidences = {categories[i]: round(prob*100,2) for i, prob in enumerate(probs)}
        
        # Sentiment
        sentiment = TextBlob(article).sentiment
        polarity = round(sentiment.polarity, 2)
        subjectivity = round(sentiment.subjectivity, 2)
        sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

        st.subheader("Predicted Category")
        st.success(predicted_category)

        st.subheader("Confidence per Category")
        for cat, conf in confidences.items():
            st.write(f"{cat}: {conf}%")

        st.subheader("Sentiment of Article")
        st.write(f"Polarity: {polarity} | Subjectivity: {subjectivity} | Sentiment: {sentiment_label}")
