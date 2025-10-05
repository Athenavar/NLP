# app.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords (only first run)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.set_page_config(
    page_title="📰 News Article Classifier",
    page_icon="📰",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📰 News Article Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter any news article to predict its category</p>", unsafe_allow_html=True)
st.markdown("---")

# Load CSV from repo
@st.cache_data
def load_data():
    df = pd.read_csv("labelled_newscatcher_dataset[1].csv")  # Must be uploaded in your repo
    df = df[['title', 'topic']]  # Keep only required columns
    return df

df = load_data()

# Preprocess
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Train model
@st.cache_data
def train_model(data):
    data['clean_text'] = data['title'].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['topic'], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=300)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

model, vectorizer = train_model(df)

# User input
user_input = st.text_area("Paste your news article here:", height=200)

if st.button("Predict Category"):
    if user_input.strip() != "":
        clean_input = preprocess(user_input)
        vector_input = vectorizer.transform([clean_input])
        prediction = model.predict(vector_input)[0]
        probs = model.predict_proba(vector_input)[0]
        categories = model.classes_

        st.markdown(f"<h3 style='color: green;'>Predicted Category: {prediction.upper()}</h3>", unsafe_allow_html=True)
        st.markdown("### Confidence per category:")
        for cat, prob in sorted(zip(categories, probs), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{cat}**")
            st.progress(int(prob * 100))
    else:
        st.warning("⚠️ Please enter some text to classify!")
