import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import altair as alt
import spacy
from googletrans import Translator

# --- Load Dataset (balanced subset: 1000 per topic) ---
@st.cache_data
def load_dataset(path, n_per_topic=1000):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    df['text'] = df['title'].fillna('')
    # Sample n_per_topic per topic
    df_balanced = df.groupby('topic', group_keys=False).apply(lambda x: x.sample(min(len(x), n_per_topic)))
    return df_balanced

df = load_dataset("labelled_newscatcher_dataset[1].csv", n_per_topic=1000)

# --- Train model ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(df['text'])
    y = df['topic']
    model = MultinomialNB()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# --- Load spaCy model ---
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()
translator = Translator()

# --- Streamlit UI ---
st.title("📰 News Classifier + Sentiment + NER")

article = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if not article.strip():
        st.warning("Please enter a news article.")
    else:
        # --- Translate if not English ---
        lang = translator.detect(article).lang
        if lang != 'en':
            article_en = translator.translate(article, dest='en').text
        else:
            article_en = article

        # --- Predict category ---
        X_input = vectorizer.transform([article_en])
        prediction = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        # --- Sentiment ---
        sentiment = TextBlob(article_en).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # --- NER ---
        doc = nlp(article_en)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # --- Display results ---
        st.subheader("Prediction Result")
        st.write(f"**Predicted Category:** {prediction}")
        st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")
        st.write("**Named Entities:**", entities if entities else "None detected")

        # --- Confidence Pie Chart ---
        df_probs = pd.DataFrame({'Category': model.classes_, 'Confidence': probs})
        chart = alt.Chart(df_probs).mark_arc().encode(
            theta=alt.Theta(field="Confidence", type="quantitative"),
            color=alt.Color(field="Category", type="nominal"),
            tooltip=['Category', 'Confidence']
        )
        st.altair_chart(chart, use_container_width=True)
