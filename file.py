import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="📰 News Article Classifier & Dashboard",
    page_icon="📰",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📰 News Article Classifier & Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter any news article to predict its category and see analytics</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("labelled_newscatcher_dataset[1].csv", sep=";", engine="python")
    df = df.dropna(subset=["title", "topic"])
    
    # Keep only the 8 main categories
    categories = ["BUSINESS", "POLITICS", "SPORTS", "TECHNOLOGY", 
                  "ENTERTAINMENT", "SCIENCE", "HEALTH", "WEATHER"]
    df = df[df["topic"].isin(categories)]
    return df

df = load_data()

# --- Preprocess ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_title'] = df['title'].apply(preprocess)

# --- Feature & Label Encoding ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_title'])
le = LabelEncoder()
y = le.fit_transform(df['topic'])

# --- Train Classifier ---
model = MultinomialNB()
model.fit(X, y)

# --- Sidebar: Show Analytics ---
st.sidebar.header("📊 Dataset Dashboard")

# Category distribution
st.sidebar.subheader("Category Distribution")
category_counts = df['topic'].value_counts()
fig, ax = plt.subplots(figsize=(6,4))
category_counts.plot(kind='bar', ax=ax, color='skyblue')
ax.set_ylabel("Number of Articles")
ax.set_xlabel("Category")
st.sidebar.pyplot(fig)

# Word clouds for each category
st.sidebar.subheader("Top Keywords per Category")
for cat in le.classes_:
    text = " ".join(df[df['topic'] == cat]['clean_title'])
    if text.strip() != "":
        wc = WordCloud(width=200, height=150, background_color='white').generate(text)
        st.sidebar.image(wc.to_array(), caption=f"{cat} Keywords", use_column_width=True)

# --- Streamlit Input ---
user_input = st.text_area("Paste your news article here:", height=200)

if st.button("Predict Category"):
    if user_input.strip() != "":
        clean_input = preprocess(user_input)
        vector_input = vectorizer.transform([clean_input])
        
        # Prediction
        prediction_idx = model.predict(vector_input)[0]
        prediction = le.inverse_transform([prediction_idx])[0]
        probs = model.predict_proba(vector_input)[0]
        categories = le.classes_

        st.markdown(f"<h3 style='color: green;'>Predicted Category: {prediction}</h3>", unsafe_allow_html=True)
        st.markdown("### Confidence per category:")
        for cat, prob in sorted(zip(categories, probs), key=lambda x: x[1], reverse=True):
            st.progress(int(prob*100))
            st.write(f"**{cat}**: {prob*100:.2f}%")
        
        # --- Top Keywords per Category ---
        feature_names = vectorizer.get_feature_names_out()
        log_probs = model.feature_log_prob_  # shape: [n_classes, n_features]

        category_keywords = {}
        for i, cat in enumerate(categories):
            contrib_scores = vector_input.toarray()[0] * log_probs[i]
            top_indices = contrib_scores.argsort()[::-1][:5]
            top_words = [feature_names[idx] for idx in top_indices if contrib_scores[idx] > 0]
            category_keywords[cat] = top_words

        st.markdown("### Top Keywords Contributing to Each Category:")
        for cat in categories:
            st.write(f"**{cat}**: {', '.join(category_keywords.get(cat, [])) if category_keywords.get(cat) else 'No keywords found'}")
        
        # --- Sentiment Analysis ---
        sentiment = TextBlob(user_input).sentiment.polarity
        if sentiment > 0.05:
            sentiment_label = "Positive"
        elif sentiment < -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        st.markdown(f"### Sentiment of Input Article: **{sentiment_label}** (Polarity: {sentiment:.2f})")

        # --- Related Articles ---
        st.markdown("### Related Articles from Dataset:")
        category_df = df[df['topic'] == prediction]
        category_vectors = vectorizer.transform(category_df['clean_title'])
        sims = cosine_similarity(vector_input, category_vectors)[0]
        top_indices = sims.argsort()[::-1][:3]  # top 3 related articles
        for idx in top_indices:
            st.write(f"- {category_df.iloc[idx]['title']} (Source: {category_df.iloc[idx]['domain']})")
        
    else:
        st.warning("⚠️ Please enter some text to classify!")
