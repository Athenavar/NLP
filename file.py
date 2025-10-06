import streamlit as st
import pandas as pd
from transformers import pipeline
from textblob import TextBlob

# --- Load your dataset ---
# Here, using a placeholder CSV, you can replace with Newscatcher dataset
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

df = load_dataset("newscatcher_dataset.csv")  # replace with your dataset path

# --- BERT Classifier via Transformers (TF backend) ---
@st.cache_resource
def get_classifier():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", framework="tf")

classifier = get_classifier()

# --- Streamlit UI ---
st.title("📰 News Article Classifier")
article = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter an article.")
    else:
        # Category prediction
        result = classifier(article)[0]
        category = result['label']  # e.g., POSITIVE/NEGATIVE (can replace with your custom trained model)
        confidence = result['score']

        # Sentiment
        sentiment = TextBlob(article).sentiment.polarity
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Display results
        st.subheader("Prediction")
        st.write(f"**Predicted Category:** {category} ({confidence*100:.2f}% confidence)")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")
