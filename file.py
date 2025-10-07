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
    try:
        df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    except FileNotFoundError:
        st.error("CSV file not found. Make sure it's in your app folder.")
        return pd.DataFrame()
    
    # Only keep the 'title' column and 'topic'
    df = df[['title', 'topic']].dropna()
    
    # Take only 1000 samples per topic to speed things up
    df = df.groupby('topic').head(1000).reset_index(drop=True)
    
    df['text'] = df['title']
    return df

df = load_dataset("labelled_newscatcher.csv")

if df.empty:
    st.stop()

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

st.subheader("Enter a news article:")
article = st.text_area("Paste your news article here:")

language_option = st.selectbox("Translate Article?", ["No", "Yes"])

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # --- Multi-language support ---
        if language_option == "Yes":
            translator = Translator()
            try:
                article = translator.translate(article, dest='en').text
            except:
                st.warning("Translation failed. Proceeding with original text.")
        
        # --- Prediction ---
        X_input = vectorizer.transform([article])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]
        
        # --- Sentiment ---
        sentiment = TextBlob(article).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        
        # --- Display ---
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")
        
        st.subheader("Prediction Confidence per Category")
        df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
        chart = alt.Chart(df_probs).mark_arc().encode(
            theta="Confidence",
            color="Category",
            tooltip=['Category', 'Confidence']
        )
        st.altair_chart(chart, use_container_width=True)
