import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

st.set_page_config(page_title="BBC News Classifier")

st.title("📰News Article Classifier")
st.write("Enter a news article to predict its category.")

# Load CSV
df = pd.read_csv("bbc-text.csv")

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess)

# Train model (on first run)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["category"], test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# User input
user_input = st.text_area("Paste your news article here:")

if st.button("Predict Category"):
    if user_input.strip() != "":
        clean_input = preprocess(user_input)
        vector_input = vectorizer.transform([clean_input])
        prediction = model.predict(vector_input)[0]
        probs = model.predict_proba(vector_input)[0]
        categories = model.classes_

        st.success(f"Predicted Category: {prediction.upper()}")
        st.markdown("### Confidence per category:")
        for cat, prob in sorted(zip(categories, probs), key=lambda x: x[1], reverse=True):
            st.write(f"{cat}: {prob*100:.2f}%")
    else:
        st.warning("Please enter some text to classify!")
