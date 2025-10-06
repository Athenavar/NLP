import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="📰 News Article Classifier & Dashboard",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📰 News Article Classifier & Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter any news article to predict its category and see analytics</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("labelled_newscatcher_dataset.csv", delimiter=';', engine='python')
    df = df.dropna(subset=['title', 'topic'])
    # Filter only 8 categories
    categories = ["BUSINESS", "POLITICS", "SPORTS", "TECHNOLOGY", "ENTERTAINMENT", "SCIENCE", "HEALTH", "WEATHER/ENVIRONMENT"]
    df = df[df['topic'].isin(categories)].reset_index(drop=True)
    return df

df = load_data()

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Preprocess titles
df['clean_title'] = df['title'].apply(preprocess)

# ----------------------------
# Vectorizer & Model
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_title'])
model = MultinomialNB()
model.fit(X, df['topic'])

# ----------------------------
# Top keywords per category
# ----------------------------
def get_top_keywords_per_category(vectorizer, model, n=10):
    top_keywords = {}
    for i, category in enumerate(model.classes_):
        top_indices = model.feature_log_prob_[i].argsort()[::-1][:n]
        top_words = [vectorizer.get_feature_names_out()[j] for j in top_indices]
        top_keywords[category] = top_words
    return top_keywords

top_keywords = get_top_keywords_per_category(vectorizer, model, n=10)

# ----------------------------
# Sidebar - Category Distribution
# ----------------------------
st.sidebar.header("📊 Dataset Dashboard")
st.sidebar.subheader("Category Distribution")
st.sidebar.bar_chart(df['topic'].value_counts())

st.sidebar.subheader("Top Keywords per Category")
for cat, keywords in top_keywords.items():
    st.sidebar.write(f"**{cat} Keywords:** {', '.join(keywords)}")

# ----------------------------
# Input Article
# ----------------------------
user_input = st.text_area("Paste your news article here:", height=200)

if st.button("Predict Category") and user_input.strip() != "":
    clean_input = preprocess(user_input)
    vector_input = vectorizer.transform([clean_input])
    
    # Prediction
    prediction = model.predict(vector_input)[0]
    probs = model.predict_proba(vector_input)[0]
    categories = model.classes_

    st.markdown(f"<h3 style='color: green;'>Predicted Category: {prediction}</h3>", unsafe_allow_html=True)
    st.markdown("### Confidence per category:")
    for cat, prob in sorted(zip(categories, probs), key=lambda x: x[1], reverse=True):
        st.progress(int(prob * 100))
        st.write(f"{cat}: {prob*100:.2f}%")
    
    # Sentiment Analysis
    sentiment = TextBlob(user_input).sentiment
    if sentiment.polarity > 0:
        sentiment_label = "Positive"
    elif sentiment.polarity < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    st.markdown(f"**Sentiment of Input Article:** {sentiment_label} (Polarity: {sentiment.polarity:.2f})")
    
    # Related Articles (Cosine Similarity)
    similarities = cosine_similarity(vector_input, X).flatten()
    top_idx = similarities.argsort()[::-1][:5]
    st.markdown("### Related Articles from Dataset:")
    for idx in top_idx:
        st.write(f"{df.loc[idx, 'title']} (Source: {df.loc[idx, 'domain']})")
else:
    st.info("⚠️ Enter some text above and click 'Predict Category' to see results.")
