import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Streamlit page config
st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📰 BBC News Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter any BBC news article to predict its category</p>", unsafe_allow_html=True)
st.markdown("---")

# Load BBC dataset
@st.cache_data
def load_data():
    df = pd.read_csv("bbc-text.csv")  # Upload this file to Streamlit Cloud
    df = df[['text', 'category']]     # Ensure column names match CSV
    return df

df = load_data()

# Train a simple model
@st.cache_data
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['category']
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = train_model(df)

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

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
