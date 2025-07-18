import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import altair as alt
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from PIL import Image

# --- Setup ---
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="centered", page_icon="💬")

col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("ucc.png")
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

# --- About ---
with st.expander("ℹ️ About this App"):
    st.markdown("""
    This app performs sentiment analysis and allows manual assignment of LDA topic labels based on top words per topic.
    """)

# --- NLTK Downloads ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    sentiment = "😊 Positive" if polarity > 0 else "😠 Negative" if polarity < 0 else "😐 Neutral"
    opinion = "Opinion" if subjectivity > 0 else "Fact"
    return polarity, subjectivity, sentiment, opinion

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
        iterations=50,
        per_word_topics=True
    )
    return lda_model

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or TXT", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment"])
    else:
        st.error("🚨 Unsupported file format.")
        st.stop()

    text_cols = df.select_dtypes(include="object").columns.tolist()
    selected_col = st.selectbox("Select Text Column", text_cols)

    # --- Sentiment Analysis ---
    if st.button("🔍 Run Sentiment Analysis"):
        results = []
        for comment in df[selected_col].dropna():
            cleaned = clean_text(comment)
            polarity, subjectivity, sentiment, opinion = analyze_sentiment(cleaned)
            results.append({
                "Original Comment": comment,
                "Cleaned Text": cleaned,
                "Polarity": polarity,
                "Subjectivity": subjectivity,
                "Sentiment": sentiment,
                "Type": opinion
            })

        sentiment_df = pd.DataFrame(results)
        st.session_state["sentiment_df"] = sentiment_df
        st.dataframe(sentiment_df)

    # --- Topic Modeling ---
    if "sentiment_df" in st.session_state:
        sentiment_df = st.session_state["sentiment_df"]
        st.markdown("---")
        st.header("🧠 LDA Topic Modeling (Manual Topic Assignment)")

        clean_texts = sentiment_df["Cleaned Text"].tolist()
        processed_texts = prepare_gensim_data(clean_texts)

        id2word = corpora.Dictionary(processed_texts)
        id2word.filter_extremes(no_below=5, no_above=0.5)
        corpus = [id2word.doc2bow(text) for text in processed_texts]
        corpus = [doc for doc in corpus if doc]

        max_topics = min(20, len(corpus)-1, len(id2word))
        num_topics = st.slider("Select Number of Topics", 3, max_topics, 5)

        if st.button("🚀 Run LDA and Show Words per Topic"):
            lda_model = train_gensim_lda_model(corpus, id2word, num_topics)
            st.session_state["lda_model"] = lda_model
            st.session_state["corpus"] = corpus
            st.session_state["id2word"] = id2word

            # Show dictionary of top words per topic
            st.subheader("📋 Top Words Per Topic (Dictionary Style)")
            for idx, topic in lda_model.print_topics(num_words=10):
                st.write(f"**Topic {idx}:** {topic}")

            # Save raw topics to session for manual labeling
            st.session_state["raw_topics"] = lda_model.print_topics(num_words=10)

        # --- Manual Topic Assignment ---
        if "raw_topics" in st.session_state:
            st.subheader("📝 Assign Custom Topic Names Based on the Top Words")

            if "custom_topic_labels" not in st.session_state:
                st.session_state["custom_topic_labels"] = {}

            for idx, topic in st.session_state["raw_topics"]:
                default_label = st.session_state["custom_topic_labels"].get(idx, f"Topic {idx+1}")
                label = st.text_input(f"Custom name for Topic {idx+1}", default_label, key=f"custom_label_{idx}")
                st.session_state["custom_topic_labels"][idx] = label

            if st.button("✅ Finalize Topic Assignment & Visualize"):
                topic_labels = st.session_state["custom_topic_labels"]
                lda_model = st.session_state["lda_model"]
                corpus = st.session_state["corpus"]
                id2word = st.session_state["id2word"]

                # Assign topics to documents
                topic_assignments = []
                for doc in corpus:
                    topic_dist = lda_model.get_document_topics(doc)
                    if topic_dist:
                        top_topic = max(topic_dist, key=lambda x: x[1])[0]
                        topic_name = topic_labels.get(top_topic, f"Topic {top_topic+1}")
                        topic_assignments.append(topic_name)
                    else:
                        topic_assignments.append("Unassigned")

                sentiment_df["Topic"] = "Unassigned"
                non_empty_mask = sentiment_df["Cleaned Text"].str.strip() != ""
                sentiment_df.loc[non_empty_mask, "Topic"] = topic_assignments

                st.dataframe(sentiment_df)

                # --- Interactive LDA Visualization ---
                st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                vis = gensimvis.prepare(lda_model, corpus, id2word)
                html_string = pyLDAvis.prepared_data_to_html(vis)
                st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

                # --- Export Final Data ---
                st.download_button("📥 Download LDA Results", sentiment_df.to_csv(index=False), "lda_results.csv")
