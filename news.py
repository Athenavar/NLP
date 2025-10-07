# ============================================
# NEWS CLASSIFIER STREAMLIT APP
# ============================================

# STEP 0: Install required packages (run once in terminal)
# pip install streamlit transformers torch tokenizers huggingface-hub requests pandas matplotlib seaborn

# ============================================
# STEP 1: Imports & Environment Setup
# ============================================
import os
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Disable Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ============================================
# STEP 2: Load Hugging Face Model
# ============================================
@st.cache_resource(show_spinner=True)
def load_model():
    return pipeline(
        "text-classification",
        model="classla/multilingual-iptc-news-topic-classifier"
    )

st.title("📰 Live News Topic Classifier")
st.caption("Classify live news headlines into 17 IPTC topics")

classifier = load_model()

# ============================================
# STEP 3: Fetch live news
# ============================================
API_KEY = st.secrets["NEWSAPI_KEY"] if "NEWSAPI_KEY" in st.secrets else "25cec2ed9e8a44dd9bedafa0a19bff57"

def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=50&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("status") == "ok":
        articles = data["articles"]
        titles = [a["title"] for a in articles if a["title"]]
        urls = [a["url"] for a in articles if a["url"]]
        return pd.DataFrame({"title": titles, "url": urls})
    else:
        st.error("Error fetching news from NewsAPI")
        return pd.DataFrame(columns=["title", "url"])

df_news = fetch_news()
st.write(f"Fetched {len(df_news)} live headlines.")

# ============================================
# STEP 4: Classify headlines
# ============================================
@st.cache_data(show_spinner=True)
def classify_headlines(titles):
    predictions = [classifier(title)[0]['label'] for title in titles]
    return predictions

if not df_news.empty:
    df_news["predicted_topic"] = classify_headlines(df_news["title"].tolist())

# ============================================
# STEP 5: Interactive filtering
# ============================================
topic_filter = st.multiselect(
    "Filter by topic:",
    options=df_news["predicted_topic"].unique(),
    default=df_news["predicted_topic"].unique()
)

filtered_df = df_news[df_news["predicted_topic"].isin(topic_filter)]

# ============================================
# STEP 6: Display headlines table
# ============================================
st.subheader("🗞️ Classified Headlines")
for idx, row in filtered_df.iterrows():
    st.markdown(f"- [{row['title']}]({row['url']}) - **{row['predicted_topic']}**")

# ============================================
# STEP 7: Visualize topic distribution
# ============================================
st.subheader("📊 Topic Distribution")
plt.figure(figsize=(10, 5))
sns.countplot(
    data=filtered_df,
    x="predicted_topic",
    palette="coolwarm",
    order=filtered_df['predicted_topic'].value_counts().index
)
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.xlabel("Topic")
plt.tight_layout()
st.pyplot(plt)
