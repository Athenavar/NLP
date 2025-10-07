import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import altair as alt
import spacy
from spacy.cli import download as spacy_download
from googletrans import Translator

# --- Load spaCy model ---
with st.spinner("Loading NLP model..."):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

translator = Translator()

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    df['text'] = df['title'].fillna('')  # Using title only
    df['topic'] = df['topic'].fillna('Unknown')
    return df

df = load_dataset("labelled_newscatcher_dataset[1].csv")

# --- Train Model ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Streamlit UI ---
st.title("📰 News Article Classifier & Sentiment Analysis")

st.subheader("Enter a news article (multiple languages supported):")
article = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # Translate to English if needed
        try:
            translated = translator.translate(article, dest='en').text
        except Exception:
            translated = article  # fallback

        # Prediction
        X_input = vectorizer.transform([translated])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        # Sentiment
        sentiment_score = TextBlob(translated).sentiment.polarity
        sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

        # Named Entity Recognition
        doc = nlp(translated)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment_score:.2f})")

        st.subheader("Prediction Confidence per Category")
        df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
        pie = alt.Chart(df_probs).mark_arc().encode(
            theta="Confidence",
            color="Category",
            tooltip=["Category", "Confidence"]
        )
        st.altair_chart(pie, use_container_width=True)

        if entities:
            st.subheader("Named Entities (NER)")
            for text, label in entities:
                st.write(f"{text} ({label})")
        else:
            st.write("No named entities found.")
