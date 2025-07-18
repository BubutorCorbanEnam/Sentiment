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

# --- University Branding ---
col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("ucc.png")  # Ensure this file is in your project directory
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ PROFESSIONAL BACKGROUND ------------------
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy. This app allows users to analyze the sentiment of comments using natural language processing.
    
    It supports both batch analysis via CSV upload and manual typing. Results include polarity, subjectivity, sentiment type, and visual insights. Ideal for researchers, marketers, and educators.
    """)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Functions for Text Analysis ---
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
def train_gensim_lda_model(corpus, _id2word, num_topics):
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=_id2word,
        num_topics=num_topics,
        random_state=50,
        passes=5,
        iterations=50,
        per_word_topics=True
    )
    return lda_model

# --- Streamlit Application Layout ---

st.markdown("Upload your text data (CSV, Excel, or TXT) to perform sentiment analysis, generate word clouds, and discover topics using LDA.")

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
            st.error("🚨 Unsupported file format. Please upload a CSV, Excel, or TXT file.")
            st.stop()

        text_cols = df.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.warning("⚠️ No text columns found in the uploaded file. Please ensure your file contains text data.")
            st.stop()

        selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

        # Sentiment Analysis & Word Cloud
        if st.button("🔍 Run Sentiment Analysis & Word Cloud"):
            with st.spinner("Analyzing sentiment and generating word cloud..."):
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

                results_df = pd.DataFrame(results)
                st.session_state["sentiment_df"] = results_df  # Store sentiment_df in session_state

                st.subheader("🗂️ Sentiment Analysis Results")
                st.dataframe(results_df)

                st.download_button(
                    label="📥 Download Sentiment Results CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

            st.markdown("---")
            st.subheader("☁️ Word Cloud")
            all_cleaned_text = " ".join(results_df["Cleaned Text"].tolist())
            if len(all_cleaned_text.strip()) > 0:
                wc_image = generate_wordcloud(all_cleaned_text)
                st.image(wc_image.to_array(), caption="Word Cloud of Cleaned Text", use_container_width=True)
            else:
                st.info("ℹ️ Not enough cleaned text available to generate a meaningful Word Cloud.")

            st.markdown("---")
            st.subheader("📊 Sentiment Distribution")
            counts = results_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y", title="Sentiment Category"),
                y=alt.Y('Count', title="Number of Comments"),
                color=alt.Color('Sentiment', legend=None),
                tooltip=['Sentiment', 'Count']
            ).properties(
                title="Distribution of Sentiment Categories"
            )
            st.altair_chart(chart, use_container_width=True)

            st.markdown("---")
            st.subheader("🎯 Sentiment Scatter Plot")
            scatter_chart = alt.Chart(results_df).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X('Polarity', title='Polarity (Negative to Positive)'),
                y=alt.Y('Subjectivity', title='Subjectivity (Fact to Opinion)'),
                color=alt.Color('Sentiment', legend=alt.Legend(title="Sentiment")),
                tooltip=['Original Comment', 'Polarity', 'Subjectivity', 'Sentiment']
            ).interactive().properties(
                title="Polarity vs. Subjectivity of Comments"
            )
            st.altair_chart(scatter_chart, use_container_width=True)

        # LDA Topic Modeling Section
        st.markdown("---")
        st.header("🧠 Topic Modeling (Latent Dirichlet Allocation - LDA)")
        st.write("LDA helps identify underlying topics in your text data.")

        if "sentiment_df" in st.session_state:
            sentiment_df = st.session_state["sentiment_df"]

            # Use only cleaned texts for LDA
            cleaned_texts = sentiment_df["Cleaned Text"].dropna().tolist()
            processed_texts_for_lda = prepare_gensim_data(cleaned_texts)

            if not processed_texts_for_lda or len(processed_texts_for_lda) < 3:
                st.warning("⚠️ Not enough valid cleaned text for topic modeling after preprocessing.")
            else:
                id2word = corpora.Dictionary(processed_texts_for_lda)
                id2word.filter_extremes(no_below=5, no_above=0.5)
                corpus = [id2word.doc2bow(text) for text in processed_texts_for_lda]
                corpus = [doc for doc in corpus if doc]

                if len(corpus) < 3 or len(id2word) < 3:
                    st.warning("⚠️ Not enough documents or vocabulary for LDA (need at least 3).")
                else:
                    max_topics = min(len(corpus) - 1, len(id2word), 20)
                    if max_topics < 3:
                        st.warning(f"⚠️ Insufficient data for at least 3 topics. Max possible: {max_topics}")
                    else:
                        num_topics = st.slider(
                            "Select Number of Topics for LDA",
                            min_value=3,
                            max_value=max_topics,
                            value=min(5, max_topics),
                            step=1
                        )

                        if st.button("🚀 Run LDA Topic Analysis"):
                            with st.spinner(f"Training LDA model with {num_topics} topics..."):
                                try:
                                    lda_model = train_gensim_lda_model(corpus, id2word, num_topics)
                                    st.session_state["lda_model"] = lda_model
                                    st.session_state["corpus"] = corpus
                                    st.session_state["id2word"] = id2word

                                    st.markdown("### 📊 Top Words Per Topic")
                                    topic_word_strings = []
                                    for idx, topic in lda_model.print_topics(num_words=10):
                                        topic_word_strings.append((idx, topic))
                                        st.write(f"**Topic {idx}:** {topic}")

                                    st.session_state["topic_word_strings"] = topic_word_strings

                                    # Prepare mapping for manual topic assignment
                                    if "topic_names" not in st.session_state:
                                        default_names = {idx: f"Topic {idx}" for idx, _ in topic_word_strings}
                                        st.session_state["topic_names"] = default_names

                                    st.markdown("---")
                                    st.subheader("✏️ Manually Assign Names to Topics")
                                    new_topic_names = {}
                                    for idx, topic in topic_word_strings:
                                        default_name = st.session_state["topic_names"].get(idx, f"Topic {idx}")
                                        new_name = st.text_input(f"Name for Topic {idx}:", value=default_name, key=f"topic_name_{idx}")
                                        new_topic_names[idx] = new_name
                                    st.session_state["topic_names"] = new_topic_names

                                    # Assign topics to each document based on highest probability topic
                                    topics_per_doc = []
                                    for doc_bow in corpus:
                                        topic_probs = lda_model.get_document_topics(doc_bow, minimum_probability=0)
                                        topic_probs_sorted = sorted(topic_probs, key=lambda x: x[1], reverse=True)
                                        topics_per_doc.append(topic_probs_sorted[0][0])

                                    # Create a Series with the manual names
                                    topic_name_map = st.session_state["topic_names"]
                                    topic_labels = [topic_name_map.get(t, f"Topic {t}") for t in topics_per_doc]

                                    # Assign topic labels back to the sentiment_df filtered to cleaned_texts length
                                    if len(topic_labels) == len(cleaned_texts):
                                        sentiment_df.loc[sentiment_df["Cleaned Text"].notna(), "Topic"] = topic_labels
                                        st.session_state["sentiment_df"] = sentiment_df
                                    else:
                                        st.warning("⚠️ Topic assignment length mismatch; could not assign topics.")

                                    st.markdown("---")
                                    st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                                    with st.spinner("Generating visualization..."):
                                        vis = gensimvis.prepare(lda_model, corpus, id2word)
                                        html_string = pyLDAvis.prepared_data_to_html(vis)
                                        st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

                                except Exception as e:
                                    st.error(f"An error occurred during LDA modeling: {e}")

            # Show updated sentiment_df with topic labels if available
            if "Topic" in sentiment_df.columns:
                st.markdown("---")
                st.subheader("🗂️ Sentiment Analysis Results with Topic Labels")
                st.dataframe(sentiment_df)

        else:
            st.info("ℹ️ Run sentiment analysis first to prepare cleaned text for topic modeling.")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing your file: {e}")
        st.info("Please ensure your file is correctly formatted and the selected column contains text data.")
else:
    st.info("☝️ Please upload a dataset (CSV, Excel, or TXT) to begin your text analysis journey!")
