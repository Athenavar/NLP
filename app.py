import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Streamlit page config
st.set_page_config(page_title="News Article Classifier", page_icon="📰", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📰 News Article Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter a news article to predict its category</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Load CSV ------------------
@st.cache_data

def load_data():
    # Load CSV, skip bad lines
    df = pd.read_csv(
        "labelled_newscatcher_dataset[1].csv",
        engine='python',
        on_bad_lines='skip',
        encoding='utf-8'
    )
    # Keep only 1000 samples per category
    df = df.groupby('category').head(1000).reset_index(drop=True)
    return df


df = load_data()

# ------------------ Preprocessing ------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ------------------ Train Model ------------------
@st.cache_data
def train_model(data):
    data['clean_text'] = data['text'].apply(preprocess)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['clean_text'])
    y = data['category']
    
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = train_model(df)

# ------------------ Prediction ------------------
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
            st.progress(int(prob*100))
    else:
        st.warning("⚠️ Please enter some text to classify!")
