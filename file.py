import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(
        path,
        sep=';',             # semicolon-separated CSV
        engine='python',
        encoding='utf-8',
        on_bad_lines='skip'  # skip bad lines
    )
    df['text'] = df['title'].fillna('')  # use only title
    df.rename(columns={'topic':'category'}, inplace=True)
    return df

df = load_dataset("labelled_newscatcher_dataset[1].csv")

# --- TF-IDF + Naive Bayes Model Training ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['category']
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Streamlit UI ---
st.title("📰 News Article Classifier & Dashboard")

# Dataset Analytics
st.subheader("📊 Dataset Dashboard")
st.write("Category Distribution")
category_counts = df['category'].value_counts()
st.bar_chart(category_counts)

st.subheader("Top Keywords per Category")
for cat in df['category'].unique():
    text = ' '.join(df[df['category']==cat]['text'].tolist())
    tfidf = TfidfVectorizer(stop_words='english', max_features=10)
    top_keywords = tfidf.fit([text]).get_feature_names_out()
    st.write(f"**{cat} Keywords:** {', '.join(top_keywords)}")

# Article input
st.subheader("Enter any news article to predict its category and see analytics")
article = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter an article.")
    else:
        X_input = vectorizer.transform([article])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]
        prob_dict = dict(zip(model.classes_, probs))

        # Sentiment Analysis
        sentiment = TextBlob(article).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Display results
        st.subheader("Prediction")
        st.write(f"**Predicted Category:** {prediction}")
        st.write("**Confidence per category:**")
        st.write(prob_dict)
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")
