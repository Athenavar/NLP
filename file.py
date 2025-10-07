import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    df['text'] = df['title'].fillna('') + " " + df['content'].fillna('')
    return df

df = load_dataset("labelled_newscatcher_dataset[1].csv")  # replace with your dataset path

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
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Display results
        st.subheader("Prediction")
        st.write(f"**Predicted Category:** {prediction}")
        st.write("**Confidence per category:**")
        st.write(prob_dict)
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

        # Related articles
        st.subheader("Related Articles from Dataset")
        df['similarity'] = df['text'].apply(lambda x: np.dot(vectorizer.transform([x]).toarray(), X_input.toarray().T)[0][0])
        top_idx = df['similarity'].nlargest(5).index
        for idx in top_idx:
            st.write(f"{df.loc[idx, 'title']} (Source: {df.loc[idx, 'domain']})")
