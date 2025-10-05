import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Page configuration
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered",
)

# Header
st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📰 BBC News Article Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter any BBC news article to predict its category</p>", unsafe_allow_html=True)
st.markdown("---")

# Load pickled model and vectorizer
@st.cache_data
def load_model():
    with open("bbc_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("bbc_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Input text area
user_input = st.text_area("Paste your news article here:", height=200)

# Predict button
if st.button("Predict Category"):
    if user_input.strip() != "":
        clean_input = preprocess(user_input)
        vector_input = vectorizer.transform([clean_input])
        prediction = model.predict(vector_input)[0]
        probs = model.predict_proba(vector_input)[0]
        categories = model.classes_

        st.markdown(f"<h3 style='color: green;'>Predicted Category: {prediction.upper()}</h3>", unsafe_allow_html=True)
        st.markdown("### Confidence per category:")

        # Show progress bars for each category
        for cat, prob in sorted(zip(categories, probs), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{cat}**")
            st.progress(int(prob * 100))

    else:
        st.warning("⚠️ Please enter some text to classify!")
