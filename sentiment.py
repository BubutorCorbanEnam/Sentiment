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
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from PIL import Image

# Setup
st.set_page_config(page_title="UCC Sentiment & Topic Analysis", layout="centered", page_icon="💬")

# Branding
col1, col2 = st.columns([1, 8])
with col1:
    st.image("ucc.png", width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment & Topic Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Functions
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
        iterations=50,
        per_word_topics=True
    )
    return lda_model

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or TXT", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment"])
        else:
            st.error("Unsupported file format.")
            st.stop()

        text_cols = df.select_dtypes(include="object").columns.tolist()
        selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

        if st.button("🔍 Run Sentiment Analysis & Word Cloud"):
            with st.spinner("Running Sentiment Analysis..."):
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

                st.subheader("🗂️ Sentiment Analysis Results")
                st.dataframe(sentiment_df)

                st.download_button("📥 Download CSV", sentiment_df.to_csv(index=False), file_name="sentiment_results.csv")

            st.subheader("☁️ Word Cloud")
            all_text = " ".join(sentiment_df["Cleaned Text"].tolist())
            if len(all_text.strip()) > 0:
                wc_image = generate_wordcloud(all_text)
                st.image(wc_image.to_array(), caption="Word Cloud", use_container_width=True)

            st.subheader("📊 Sentiment Distribution")
            counts = sentiment_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y"),
                y='Count',
                color='Sentiment',
                tooltip=['Sentiment', 'Count']
            )
            st.altair_chart(chart)

            st.subheader("🎯 Sentiment Scatter Plot")
            scatter_chart = alt.Chart(sentiment_df).mark_circle(size=80).encode(
                x=alt.X('Polarity', title='Polarity'),
                y=alt.Y('Subjectivity', title='Subjectivity'),
                color='Sentiment',
                tooltip=['Original Comment', 'Polarity', 'Subjectivity']
            ).interactive()
            st.altair_chart(scatter_chart)

        # Topic Modeling Section
        if "sentiment_df" in st.session_state:
            st.markdown("---")
            st.header("🧠 Topic Modeling (LDA)")
            st.write("Using the cleaned comments from sentiment analysis directly.")

            cleaned_comments = st.session_state["sentiment_df"]["Cleaned Text"].dropna().tolist()
            processed_texts = prepare_gensim_data(cleaned_comments)

            id2word = corpora.Dictionary(processed_texts)
            id2word.filter_extremes(no_below=2, no_above=0.5)
            corpus = [id2word.doc2bow(text) for text in processed_texts]
            corpus = [doc for doc in corpus if doc]

            if len(corpus) < 3 or len(id2word) < 3:
                st.warning("Not enough unique words or documents for LDA.")
            else:
                num_topics = st.slider("Select Number of Topics", 3, 20, 5)

                if st.button("🚀 Run LDA Topic Modeling"):
                    lda_model = train_gensim_lda_model(corpus, id2word, num_topics)
                    st.session_state["lda_model"] = lda_model
                    st.session_state["corpus"] = corpus
                    st.session_state["id2word"] = id2word

                    st.subheader("Top Words Per Topic")
                    topic_labels = {}
                    for idx, topic in lda_model.print_topics(num_words=10):
                        st.write(f"**Topic {idx+1}:** {topic}")
                        topic_labels[idx] = st.text_input(f"Label for Topic {idx+1}", f"Topic {idx+1}")

                    if st.button("✅ Finalize Topic Assignment & Visualize"):
                        st.session_state["topic_names"] = topic_labels

                        topic_assignments = []
                        for doc in corpus:
                            topic_dist = lda_model.get_document_topics(doc)
                            if topic_dist:
                                top_topic = max(topic_dist, key=lambda x: x[1])[0]
                                topic_name = st.session_state["topic_names"].get(top_topic, f"Topic {top_topic+1}")
                                topic_assignments.append(topic_name)
                            else:
                                topic_assignments.append("Unassigned")

                        # Safe assignment to match dataframe shape
                        st.session_state["sentiment_df"]["Topic"] = "Unassigned"
                        non_empty_mask = st.session_state["sentiment_df"]["Cleaned Text"].str.strip() != ""
                        st.session_state["sentiment_df"].loc[non_empty_mask, "Topic"] = topic_assignments

                        st.subheader("🗂️ Sentiment & Topic Results")
                        st.dataframe(st.session_state["sentiment_df"])

                        # Interactive Visualization
                        st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                        vis = gensimvis.prepare(lda_model, corpus, id2word)
                        html_string = pyLDAvis.prepared_data_to_html(vis)
                        st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("☝️ Please upload a dataset to begin.")
