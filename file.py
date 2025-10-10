import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import spacy

# -------------------------------
# Load English NER model
# -------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("spaCy English model not found. Run `python -m spacy download en_core_web_sm`.")
    st.stop()

# -------------------------------
# Load and train model
# -------------------------------
@st.cache_data
def load_dataset(path):
    try:
        df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path, on_bad_lines='skip')

    if 'title' not in df.columns or 'topic' not in df.columns:
        st.error("CSV must contain 'title' and 'topic' columns.")
        return pd.DataFrame()

    df['text'] = df['title'].fillna('')
    return df

@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

# -------------------------------
# Perform NER
# -------------------------------
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="News Analyzer", page_icon="🧠", layout="centered")
st.title("🧠 News Category, Sentiment & NER (English Only)")

st.write("Analyze an English news article for its category, sentiment, and named entities.")

# Dataset loading
csv_path = st.text_input("Enter your dataset path:", "labelled_newscatcher_dataset[1].csv")

df = load_dataset(csv_path)
if not df.empty:
    vectorizer, model = train_model(df)
else:
    st.stop()

# User input
article = st.text_area("📰 Paste your news article here:", height=200)

if st.button("Analyze"):
    if article.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # -----------------------------
        # 1️⃣ Category Prediction
        # -----------------------------
        X_input = vectorizer.transform([article])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        st.subheader("📂 Predicted Category")
        st.write(f"**Category:** {prediction}")

        # Confidence chart
        df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
        chart = alt.Chart(df_probs).mark_bar().encode(
            x='Category',
            y='Confidence',
            color='Category'
        )
        st.altair_chart(chart, use_container_width=True)

        # -----------------------------
        # 2️⃣ Sentiment Analysis
        # -----------------------------
        sentiment = TextBlob(article).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        st.subheader("💬 Sentiment")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

        # -----------------------------
        # 3️⃣ Named Entity Recognition
        # -----------------------------
        st.subheader("🧩 Named Entities (NER)")
        entities = perform_ner(article)
        if entities:
            df_entities = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(df_entities, use_container_width=True)
        else:
            st.write("No entities found in the text.")
