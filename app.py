import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import nltk, re

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Page config
st.set_page_config(page_title="News Classifier", page_icon="📰", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📰 News Article Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter any news article to predict its category</p>", unsafe_allow_html=True)
st.markdown("---")

# Dataset selection
dataset_option = st.selectbox("Choose Dataset:", ["BBC News", "20 Newsgroups"])

@st.cache_data
def load_data(dataset_name):
    if dataset_name == "BBC News":
        df = pd.read_csv("bbc-text.csv")
        X = df['text']
        y = df['category']
        categories = sorted(df['category'].unique())
    else:
        newsgroups = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
        X = newsgroups.data
        y = newsgroups.target
        categories = newsgroups.target_names
    return X, y, categories

X, y, categories = load_data(dataset_option)

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

X_clean = [preprocess(doc) for doc in X]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(X_clean)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_tfidf, y)

# User input
user_input = st.text_area("Paste your news article here:", height=200)

# Predict button
if st.button("Predict Category"):
    if user_input.strip() != "":
        clean_input = preprocess(user_input)
        vector_input = vectorizer.transform([clean_input])
        prediction = model.predict(vector_input)[0]
        probs = model.predict_proba(vector_input)[0]

        # Map prediction to category names if 20 Newsgroups
        if dataset_option == "20 Newsgroups":
            prediction_name = categories[prediction]
        else:
            prediction_name = prediction

        st.markdown(f"<h3 style='color: green;'>Predicted Category: {prediction_name.upper()}</h3>", unsafe_allow_html=True)
        st.markdown("### Confidence per category:")

        # Show progress bars
        for cat, prob in sorted(zip(categories, probs), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{cat}**")
            st.progress(int(prob * 100))
    else:
        st.warning("⚠️ Please enter some text to classify!")
