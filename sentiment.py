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
    logo = Image.open("ucc.png")  # Put your logo file in the same directory
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

with st.expander("ℹ️ About this App"):
    st.markdown("""
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy. This app allows users to analyze the sentiment of comments using natural language processing.
    
    It supports batch analysis via CSV upload and manual typing. Results include polarity, subjectivity, sentiment type, word clouds, and topic modeling with manual topic naming.
    """)

# NLTK Downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Text Cleaning ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

# --- Sentiment Analysis ---
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    if polarity > 0:
        sentiment = "😊 Positive"
    elif polarity < 0:
        sentiment = "😠 Negative"
    else:
        sentiment = "😐 Neutral"
    opinion = "Opinion" if subjectivity > 0 else "Fact"
    return polarity, subjectivity, sentiment, opinion

# --- Word Cloud ---
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words)
    return wc.generate(text)

# --- Prepare Text for LDA ---
def prepare_gensim_data(texts):
    custom_stopwords = stop_words.union({'from', 'subject', 're', 'edu', 'use'})
    processed_texts = [
        [word for word in simple_preprocess(str(doc), deacc=True) if word not in custom_stopwords]
        for doc in texts
    ]
    return processed_texts

# --- Train LDA ---
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
    if not text_cols:
        st.warning("⚠️ No text columns found.")
        st.stop()

    selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

    # --- Sentiment Analysis and Word Cloud ---
    if st.button("🔍 Run Sentiment Analysis & Word Cloud"):
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

        # Show results table
        st.subheader("🗂️ Sentiment Analysis Results")
        st.dataframe(sentiment_df)

        # Download button
        st.download_button(
            label="📥 Download Sentiment Results CSV",
            data=sentiment_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )

        # Word Cloud
        st.markdown("---")
        st.subheader("☁️ Word Cloud")
        all_cleaned_text = " ".join(sentiment_df["Cleaned Text"].tolist())
        if len(all_cleaned_text.strip()) > 0:
            wc_image = generate_wordcloud(all_cleaned_text)
            st.image(wc_image.to_array(), caption="Word Cloud of Cleaned Text", use_container_width=True)
        else:
            st.info("ℹ️ Not enough cleaned text for word cloud.")

        # Sentiment Distribution Chart (Bar Plot)
        st.markdown("---")
        st.subheader("📊 Sentiment Distribution")
        counts = sentiment_df['Sentiment'].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        bar_chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X('Sentiment', sort="-y", title="Sentiment Category"),
            y=alt.Y('Count', title="Number of Comments"),
            color=alt.Color('Sentiment', legend=None),
            tooltip=['Sentiment', 'Count']
        ).properties(title="Distribution of Sentiment Categories")
        st.altair_chart(bar_chart, use_container_width=True)

        # Sentiment Scatter Plot (Polarity vs Subjectivity)
        st.markdown("---")
        st.subheader("🎯 Sentiment Scatter Plot")
        scatter_chart = alt.Chart(sentiment_df).mark_circle(size=80, opacity=0.7).encode(
            x=alt.X('Polarity', title='Polarity (Negative to Positive)'),
            y=alt.Y('Subjectivity', title='Subjectivity (Fact to Opinion)'),
            color=alt.Color('Sentiment', legend=alt.Legend(title="Sentiment")),
            tooltip=['Original Comment', 'Polarity', 'Subjectivity', 'Sentiment']
        ).interactive().properties(title="Polarity vs. Subjectivity of Comments")
        st.altair_chart(scatter_chart, use_container_width=True)

    # --- LDA Topic Modeling ---
    if "sentiment_df" in st.session_state:
        sentiment_df = st.session_state["sentiment_df"]
        st.markdown("---")
        st.header("🧠 Topic Modeling (Latent Dirichlet Allocation - LDA)")
        clean_texts = sentiment_df["Cleaned Text"].tolist()
        processed_texts = prepare_gensim_data(clean_texts)

        id2word = corpora.Dictionary(processed_texts)
        id2word.filter_extremes(no_below=5, no_above=0.5)
        corpus = [id2word.doc2bow(text) for text in processed_texts]
        corpus = [doc for doc in corpus if doc]

        max_topics = min(20, len(corpus) - 1, len(id2word))
        if max_topics < 3:
            st.warning(f"⚠️ Not enough data to generate topics (need at least 3). You have {max_topics}.")
        else:
            num_topics = st.slider(
                "Select Number of Topics for LDA (More topics can be harder to interpret)",
                min_value=3,
                max_value=max_topics,
                value=min(5, max_topics),
                step=1,
            )

            if st.button("🚀 Run LDA Topic Analysis"):
                lda_model = train_gensim_lda_model(corpus, id2word, num_topics)
                st.session_state["lda_model"] = lda_model
                st.session_state["corpus"] = corpus
                st.session_state["id2word"] = id2word

                st.subheader("📋 Top Words Per Topic (Dictionary Style)")
                raw_topics = lda_model.print_topics(num_words=10)
                st.session_state["raw_topics"] = raw_topics
                for idx, topic in raw_topics:
                    st.write(f"**Topic {idx}:** {topic}")

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

                    st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                    vis = gensimvis.prepare(lda_model, corpus, id2word)
                    html_string = pyLDAvis.prepared_data_to_html(vis)
                    st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

                    st.download_button(
                        "📥 Download LDA Results",
                        sentiment_df.to_csv(index=False),
                        "lda_results.csv"
                    )

else:
    st.info("☝️ Please upload a dataset (CSV, Excel, or TXT) to begin your text analysis journey!")
