import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import altair as alt
from deep_translator import GoogleTranslator
import requests

# -------------------------------
# LOAD DATASET & TRAIN MODEL
# -------------------------------
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    if 'title' in df.columns and 'topic' in df.columns:
        df['text'] = df['title'].fillna('')
        return df
    else:
        st.error("CSV must contain 'title' and 'topic' columns.")
        return pd.DataFrame()

@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

# -------------------------------
# SIMPLE RULE-BASED NER FUNCTION
# -------------------------------
def simple_ner(text):
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
    entity_types = []
    for e in entities:
        if any(word in e.lower() for word in ["inc", "corp", "university", "organization", "ltd"]):
            entity_types.append(("ORG", e))
        elif any(word in e.lower() for word in ["india", "usa", "china", "russia", "germany", "france"]):
            entity_types.append(("GPE", e))
        else:
            entity_types.append(("PERSON", e))
    return entity_types

# -------------------------------
# TRANSLATION
# -------------------------------
def translate_text(text, src_lang):
    if src_lang != "en":
        try:
            return GoogleTranslator(source=src_lang, target="en").translate(text)
        except Exception:
            return text
    return text

# -------------------------------
# NEWS FETCHER
# -------------------------------
def fetch_news(query, api_key, lang="en", limit=3):
    url = f"https://newsapi.org/v2/everything?q={query}&language={lang}&pageSize={limit}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    return [a["title"] + " " + (a.get("description") or "") for a in data.get("articles", [])]

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("🧠 AI News Intelligence Suite (No Torch Version)")

option = st.sidebar.selectbox(
    "Choose Mode:",
    ["📰 News Classifier & Sentiment", "🌍 Multilingual NER"]
)

# =====================================================
# OPTION 1: CLASSIFIER + SENTIMENT
# =====================================================
if option == "📰 News Classifier & Sentiment":
    st.header("📰 News Article Classifier & Sentiment Analysis")

    df = load_dataset("labelled_newscatcher_dataset[1].csv")
    if not df.empty:
        vectorizer, model = train_model(df)

        article = st.text_area("Paste your news article here:")

        if st.button("Predict"):
            if article.strip() == "":
                st.warning("Please enter a news article.")
            else:
                X_input = vectorizer.transform([article])
                prediction = model.predict(X_input)[0]
                probs = model.predict_proba(X_input)[0]

                # Sentiment
                sentiment = TextBlob(article).sentiment.polarity
                sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

                st.subheader("Prediction Result")
                st.write(f"**Predicted Category:** {prediction}")
                st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

                # Confidence Chart
                df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
                chart = alt.Chart(df_probs).mark_bar().encode(
                    x='Category',
                    y='Confidence',
                    color='Category'
                )
                st.altair_chart(chart, use_container_width=True)

# =====================================================
# OPTION 2: MULTILINGUAL NER
# =====================================================
elif option == "🌍 Multilingual NER":
    st.header("🌍 Multilingual Named Entity Recognition (No Torch)")

    api_key = st.text_input("🔑 Enter your NewsAPI key:")
    query = st.text_input("Search news topic:", "AI")
    lang = st.selectbox("Select language:", ["en", "hi", "fr", "es", "de"])

    if st.button("Run NER"):
        if not api_key:
            st.error("Please provide your NewsAPI key.")
        else:
            with st.spinner("Fetching and analyzing news..."):
                articles = fetch_news(query, api_key, lang)
                for i, art in enumerate(articles, 1):
                    st.subheader(f"📰 Article {i}")
                    st.write(art[:200] + "...")

                    text_en = translate_text(art, lang)
                    entities = simple_ner(text_en)

                    if entities:
                        df_entities = pd.DataFrame(entities, columns=["Entity Type", "Entity"])
                        st.dataframe(df_entities)
                    else:
                        st.write("No named entities found.")
