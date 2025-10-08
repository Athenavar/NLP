import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import altair as alt

# Translation
try:
    from deep_translator import GoogleTranslator
    translator_available = True
except:
    translator_available = False

# --- Load Dataset ---
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    df['text'] = df['title'].fillna('')  # use title only
    # Take only 1000 articles per category for speed
    df = df.groupby('topic').head(1000).reset_index(drop=True)
    return df

df = load_dataset("labelled_newscatcher_dataset[1].csv")

# --- TF-IDF + Naive Bayes Model Training ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Streamlit UI ---
st.title("📰 News Article Classifier & Sentiment Analysis with Translation")

# Input article
article = st.text_area("Paste your news article here:")

# Translation option
if translator_available:
    translate_option = st.selectbox("Translate article before prediction?", ["No", "Yes"])
else:
    translate_option = "No"

if st.button("Predict"):
    if article.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # --- Translate if enabled ---
        if translate_option == "Yes":
            try:
                article_translated = GoogleTranslator(source='auto', target='en').translate(article)
                st.subheader("Translated Article")
                st.write(article_translated)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                article_translated = article
        else:
            article_translated = article

        # --- Prediction ---
        X_input = vectorizer.transform([article_translated])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        # Sentiment analysis
        sentiment = TextBlob(article_translated).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

        st.subheader("Prediction Confidence per Category")
        df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
        chart = alt.Chart(df_probs).mark_bar().encode(
            x='Category',
            y='Confidence',
            color='Category'
        )
        st.altair_chart(chart, use_container_width=True)

        # --- Simple Keyword Highlighting ---
        st.subheader("Important Keywords (Top TF-IDF)")
        feature_names = vectorizer.get_feature_names_out()
        X_vectorized = vectorizer.transform([article_translated])
        top_indices = X_vectorized.toarray()[0].argsort()[-10:][::-1]
        top_keywords = [feature_names[i] for i in top_indices if X_vectorized[0,i] > 0]
        st.write(top_keywords if top_keywords else "No significant keywords found.")
