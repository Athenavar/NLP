import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import altair as alt  # altair is preinstalled in Streamlit Cloud

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    df['text'] = df['title'].fillna('') + " " + df.get('content', '').fillna('')
    return df

df = load_dataset("labelled_newscatcher_dataset[1].csv")  # Replace with your CSV path

# --- TF-IDF + Naive Bayes Model Training ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']  # using 'topic' as category column
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Streamlit UI ---
st.title("📰 News Article Classifier & Sentiment Analysis")

st.subheader("Enter a news article:")
article = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # Transform and predict
        X_input = vectorizer.transform([article])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]
        prob_dict = dict(zip(model.classes_, probs))

        # Sentiment analysis
        sentiment = TextBlob(article).sentiment.polarity
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Display prediction
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

        # Confidence bar chart using Altair
        st.subheader("Prediction Confidence per Category")
        df_probs = pd.DataFrame({
            'Category': model.classes_,
            'Confidence': probs
        })
        chart = alt.Chart(df_probs).mark_bar().encode(
            x='Category',
            y='Confidence',
            color='Category'
        )
        st.altair_chart(chart, use_container_width=True)
