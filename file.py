import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import plotly.express as px

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    # Your dataset separator is ';'
    df = pd.read_csv(path, sep=';', error_bad_lines=False)
    df['text'] = df['title'].fillna('') + " " + df.get('content', '').fillna('')  # 'content' may not exist
    return df

df = load_dataset("labelled_newscatcher_dataset[1].csv")  # Replace with your dataset path

# --- TF-IDF + Naive Bayes Model Training ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']  # Column name in your CSV is 'topic'
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Streamlit UI ---
st.title("📰 News Article Classifier & Dashboard")

st.subheader("Enter any news article to predict its category and sentiment")
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
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

        # Confidence Pie Chart
        st.subheader("Confidence per Category")
        pie_df = pd.DataFrame({
            'Category': model.classes_,
            'Confidence': probs
        })
        fig = px.pie(pie_df, names='Category', values='Confidence', title='Confidence Distribution')
        st.plotly_chart(fig)

        # Optional: Show raw confidence values
        st.write("**Confidence Scores:**")
        st.json(prob_dict)
