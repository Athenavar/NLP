import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from googletrans import Translator
import altair as alt

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    df['text'] = df['title'].fillna('')
    
    # Take only 1000 samples per topic to reduce load time
    df_subset = df.groupby('topic').head(1000).reset_index(drop=True)
    return df_subset

df = load_dataset("labelled_newscatcher_dataset[1].csv")

# --- TF-IDF + Naive Bayes Model Training ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Translator ---
translator = Translator()

# --- Streamlit UI ---
st.title("📰 News Article Classifier & Sentiment Analysis")

st.subheader("Enter a news article:")
article = st.text_area("Paste your news article here:")

# Language selection
lang = st.selectbox("Translate to English if needed:", ["auto", "en", "fr", "es", "de", "hi"])

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # Translate if needed
        if lang != "auto":
            try:
                article_translated = translator.translate(article, dest='en').text
            except:
                article_translated = article
        else:
            article_translated = article

        X_input = vectorizer.transform([article_translated])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        # Sentiment analysis
        sentiment = TextBlob(article_translated).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Display prediction
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

        # Confidence per category
        st.subheader("Prediction Confidence per Category")
        df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
        chart = alt.Chart(df_probs).mark_bar().encode(
            x='Category',
            y='Confidence',
            color='Category'
        )
        st.altair_chart(chart, use_container_width=True)
