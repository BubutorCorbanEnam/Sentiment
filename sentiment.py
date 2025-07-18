import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
import altair as alt
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from PIL import Image

# Setup
st.set_page_config(page_title="UCC Sentiment & Topic Analysis", layout="centered", page_icon="💬")

col1, col2 = st.columns([1,8])
with col1:
    st.image("ucc.png", width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment & Topic Modeling App</h4>", unsafe_allow_html=True)

st.markdown("---")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Text Processing Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity,3)
    subjectivity = round(blob.sentiment.subjectivity,3)
    sentiment = "😊 Positive" if polarity > 0 else "😠 Negative" if polarity < 0 else "😐 Neutral"
    return polarity, subjectivity, sentiment

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words)
    return wc.generate(text)

def prepare_gensim_data(texts):
    custom_stopwords = stop_words.union({'from', 'subject', 're', 'edu', 'use'})
    processed_texts = [
        [word for word in simple_preprocess(str(doc), deacc=True) if word not in custom_stopwords]
        for doc in texts
    ]
    return processed_texts

@st.cache_resource(show_spinner=False)
def train_gensim_lda_model(_corpus, _id2word, num_topics):
    lda_model = gensim.models.LdaModel(
        corpus=_corpus,
        id2word=_id2word,
        num_topics=num_topics,
        random_state=50,
        passes=5,
        per_word_topics=True
    )
    return lda_model

# --- App Workflow ---
uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or TXT", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file) if uploaded_file.name.endswith((".xlsx", ".xls")) else pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment"])
    
    text_cols = df.select_dtypes(include="object").columns.tolist()
    selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

    if st.button("🔍 Run Sentiment Analysis"):
        results = []
        for comment in df[selected_col].dropna():
            cleaned = clean_text(comment)
            polarity, subjectivity, sentiment = analyze_sentiment(cleaned)
            results.append({
                "Original": comment,
                "Cleaned Text": cleaned,
                "Polarity": polarity,
                "Subjectivity": subjectivity,
                "Sentiment": sentiment
            })
        
        sentiment_df = pd.DataFrame(results)
        st.session_state["sentiment_df"] = sentiment_df

        # Wordcloud
        st.subheader("☁️ Word Cloud")
        all_text = " ".join(sentiment_df["Cleaned Text"].tolist())
        wc_image = generate_wordcloud(all_text)
        st.image(wc_image.to_array(), caption="Word Cloud", use_container_width=True)

        # Sentiment Distribution Bar Plot
        st.subheader("📊 Sentiment Distribution")
        counts = sentiment_df['Sentiment'].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X('Sentiment', sort="-y"),
            y='Count',
            color='Sentiment',
            tooltip=['Sentiment', 'Count']
        ).properties(width=600)
        st.altair_chart(chart)

        # Scatter Plot
        st.subheader("🎯 Sentiment Scatter Plot")
        scatter = alt.Chart(sentiment_df).mark_circle(size=80, opacity=0.7).encode(
            x=alt.X('Polarity', title='Polarity'),
            y=alt.Y('Subjectivity', title='Subjectivity'),
            color='Sentiment',
            tooltip=['Original', 'Polarity', 'Subjectivity']
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)

# --- Topic Modeling Section ---
if "sentiment_df" in st.session_state:
    sentiment_df = st.session_state["sentiment_df"]

    st.markdown("---")
    st.header("🧠 LDA Topic Modeling (Manual Topic Assignment Supported)")

    cleaned_texts = sentiment_df["Cleaned Text"].tolist()
    processed_texts = prepare_gensim_data(cleaned_texts)

    id2word = corpora.Dictionary(processed_texts)
    id2word.filter_extremes(no_below=5, no_above=0.5)

    corpus = [id2word.doc2bow(text) for text in processed_texts]

    # Get valid document indices
    valid_indices = [i for i, doc in enumerate(corpus) if len(doc) > 0]
    corpus = [corpus[i] for i in valid_indices]

    if len(corpus) >= 3 and len(id2word) >= 3:
        num_topics = st.slider("Select Number of Topics", 3, 20, 5)

        if st.button("🚀 Run LDA Topic Modeling"):
            lda_model = train_gensim_lda_model(corpus, id2word, num_topics)

            st.subheader("🔍 Top Words Per Topic")
            topic_words = {}
            for idx, topic in lda_model.show_topics(formatted=False):
                words = ", ".join([w for w, p in topic])
                topic_words[idx] = words
                st.write(f"**Topic {idx}**: {words}")

            # Manual Topic Assignment
            if "topic_labels" not in st.session_state:
                st.session_state["topic_labels"] = [""] * num_topics

            with st.form("manual_topic_assignment"):
                st.markdown("### ✍️ Manually Assign Topic Names")
                for i in range(num_topics):
                    st.session_state["topic_labels"][i] = st.text_input(
                        f"Topic {i}: {topic_words[i]}",
                        value=st.session_state["topic_labels"][i]
                    )
                submitted = st.form_submit_button("Save Topics")

            # Assign topics to documents
            topic_assignments = []
            for bow in corpus:
                topic_probs = lda_model.get_document_topics(bow)
                if topic_probs:
                    top_topic = max(topic_probs, key=lambda x: x[1])[0]
                    topic_assignments.append(st.session_state["topic_labels"][top_topic])
                else:
                    topic_assignments.append("Unassigned")

            # Map back to sentiment_df using valid_indices
            sentiment_df["Topic"] = "Unassigned"
            for idx_in_corpus, doc_idx in enumerate(valid_indices):
                sentiment_df.at[doc_idx, "Topic"] = topic_assignments[idx_in_corpus]

            st.session_state["sentiment_df"] = sentiment_df

            # Show updated dataframe
            st.subheader("📋 Sentiment + Topic Data")
            st.dataframe(sentiment_df)

            # Interactive pyLDAvis
            st.subheader("📈 Interactive LDA Visualization")
            with st.spinner("Generating interactive visualization..."):
                vis = gensimvis.prepare(lda_model, corpus, id2word)
                html_string = pyLDAvis.prepared_data_to_html(vis)
                st.components.v1.html(html_string, width=1000, height=800, scrolling=True)
    else:
        st.warning("⚠️ Not enough data after preprocessing for LDA topic modeling.")
