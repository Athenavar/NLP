import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import altair as alt
import spacy
from googletrans import Translator

# Load English spaCy model
nlp = spacy.load("en_core_web_sm")
translator = Translator()

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    df['text'] = df['title'].fillna('')  
    return df

df = load_dataset("labelled_newscatcher_dataset[1].csv")

# --- TF-IDF + Naive Bayes Model Training ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']  # 'topic' is the category column
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Streamlit UI ---
st.title("📰 News Article Classifier & Sentiment Analysis with NER & Multi-language Support")

st.subheader("Enter a news article:")
article = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # --- Multi-language support ---
        try:
            detected_lang = translator.detect(article).lang
            if detected_lang != 'en':
                article_en = translator.translate(article, dest='en').text
            else:
                article_en = article
        except:
            article_en = article  # fallback in case translation fails

        # TF-IDF + Naive Bayes prediction
        X_input = vectorizer.transform([article_en])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        # Sentiment analysis
        sentiment = TextBlob(article_en).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Named Entity Recognition
        doc = nlp(article_en)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # --- Display results ---
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

        st.subheader("Named Entities Detected")
        if entities:
            for ent_text, ent_label in entities:
                st.write(f"{ent_text} ({ent_label})")
        else:
            st.write("No named entities detected.")

        # Confidence per category
        st.subheader("Prediction Confidence per Category")
        df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
        chart = alt.Chart(df_probs).mark_bar().encode(
            x='Category',
            y='Confidence',
            color='Category'
        )
        st.altair_chart(chart, use_container_width=True)
