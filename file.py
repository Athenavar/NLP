import streamlit as st
import pandas as pd
import numpy as np
import spacy
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import altair as alt
from deep_translator import GoogleTranslator

# -------------------------------
# DATASET LOADING & MODEL TRAINING
# -------------------------------
@st.cache_data
def load_dataset(path):
    try:
        df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path)  # fallback for normal CSV
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
# LOAD SPACY MODELS
# -------------------------------
@st.cache_resource
def load_spacy_models():
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except:
        st.error("⚠️ English model missing. Run: `python -m spacy download en_core_web_sm`")
        nlp_en = None

    try:
        nlp_multi = spacy.load("xx_ent_wiki_sm")
    except:
        st.warning("🌍 Multilingual model missing. Run: `python -m spacy download xx_ent_wiki_sm`")
        nlp_multi = None

    return nlp_en, nlp_multi

nlp_en, nlp_multi = load_spacy_models()

# -------------------------------
# TRANSLATION FUNCTION
# -------------------------------
def translate_text(text, src_lang):
    if src_lang != "en":
        try:
            return GoogleTranslator(source=src_lang, target="en").translate(text)
        except Exception:
            return text
    return text

# -------------------------------
# FETCH NEWS USING NEWSAPI
# -------------------------------
def fetch_news(query, api_key, lang="en", limit=3):
    url = f"https://newsapi.org/v2/everything?q={query}&language={lang}&pageSize={limit}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data.get("status") != "ok":
        st.error(f"Error: {data.get('message', 'Unknown error')}")
        return []
    return [a["title"] + " " + (a.get("description") or "") for a in data.get("articles", [])]

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.set_page_config(page_title="AI News Suite", page_icon="🧠", layout="wide")
st.title("🧠 AI News Intelligence Suite")

mode = st.sidebar.radio("Choose Mode:", ["📰 News Classifier + Sentiment", "🌍 Multilingual NER"])

# =====================================================
# MODE 1: NEWS CLASSIFIER + SENTIMENT
# =====================================================
if mode == "📰 News Classifier + Sentiment":
    st.header("📰 News Article Classifier & Sentiment Analysis")

    df = load_dataset("labelled_newscatcher_dataset[1].csv")
    if not df.empty:
        vectorizer, model = train_model(df)

        article = st.text_area("Paste your news article here:")

        if st.button("🔍 Predict"):
            if article.strip() == "":
                st.warning("Please enter a news article.")
            else:
                X_input = vectorizer.transform([article])
                prediction = model.predict(X_input)[0]
                probs = model.predict_proba(X_input)[0]

                # Sentiment Analysis
                sentiment = TextBlob(article).sentiment.polarity
                sentiment_label = (
                    "Positive" if sentiment > 0 else
                    "Negative" if sentiment < 0 else
                    "Neutral"
                )

                st.subheader("📊 Prediction Result")
                st.write(f"**Predicted Category:** {prediction}")
                st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

                # Confidence Chart
                st.subheader("🔎 Prediction Confidence per Category")
                df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
                chart = alt.Chart(df_probs).mark_bar().encode(
                    x='Category',
                    y='Confidence',
                    color='Category'
                )
                st.altair_chart(chart, use_container_width=True)

# =====================================================
# MODE 2: MULTILINGUAL NER
# =====================================================
elif mode == "🌍 Multilingual NER":
    st.header("🌍 Multilingual Named Entity Recognition")

    api_key = st.text_input("🔑 Enter your NewsAPI key:")
    query = st.text_input("Search news topic:", "AI")
    lang = st.selectbox("Select article language:", ["en", "hi", "fr", "es", "de", "zh", "ru"])

    if st.button("🚀 Run NER"):
        if not api_key:
            st.error("Please enter your NewsAPI key.")
        else:
            with st.spinner("Fetching and analyzing news..."):
                articles = fetch_news(query, api_key, lang)

                if not articles:
                    st.warning("No articles found. Try another topic or language.")
                else:
                    for i, article in enumerate(articles, 1):
                        st.subheader(f"📰 Article {i}")
                        st.write(article)

                        # Translate if not English
                        text_en = translate_text(article, lang)
                        st.caption("🌐 Translated (English):")
                        st.write(text_en)

                        # NER detection
                        if lang == "en" and nlp_en:
                            doc = nlp_en(text_en)
                        elif nlp_multi:
                            doc = nlp_multi(text_en)
                        else:
                            st.error("No spaCy model loaded.")
                            continue

                        entities = [(ent.text, ent.label_) for ent in doc.ents]
                        if entities:
                            df_entities = pd.DataFrame(entities, columns=["Entity", "Entity Type"])
                            st.dataframe(df_entities)
                        else:
                            st.info("No named entities found.")

# -------------------------------
# END OF APP
# -------------------------------
