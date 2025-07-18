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

# Setup
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="centered", page_icon="💬")

# University Branding
col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("ucc.png")
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

# About
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy.
    This app performs sentiment analysis, word cloud generation, and topic modeling (LDA).
    """)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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
    if polarity > 0:
        sentiment = "😊 Positive"
    elif polarity < 0:
        sentiment = "😠 Negative"
    else:
        sentiment = "😐 Neutral"
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
            st.error("🚨 Unsupported file format.")
            st.stop()

        text_cols = df.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.warning("⚠️ No text columns found.")
            st.stop()

        selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

        # Run Sentiment Analysis & Word Cloud
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
            st.session_state["sentiment_df"] = sentiment_df  # Save for use in topic modeling

            st.subheader("🗂️ Sentiment Analysis Results")
            st.dataframe(sentiment_df)

            st.download_button(
                label="📥 Download Sentiment Results CSV",
                data=sentiment_df.to_csv(index=False).encode('utf-8'),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

            # Word Cloud
            st.markdown("---")
            st.subheader("☁️ Word Cloud")
            all_text = " ".join(sentiment_df["Cleaned Text"].tolist())
            if all_text.strip():
                wc_image = generate_wordcloud(all_text)
                st.image(wc_image.to_array(), caption="Word Cloud of Cleaned Text", use_container_width=True)
            else:
                st.info("ℹ️ Not enough text to generate Word Cloud.")

            # Sentiment Distribution
            st.markdown("---")
            st.subheader("📊 Sentiment Distribution")
            counts = sentiment_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y", title="Sentiment Category"),
                y=alt.Y('Count', title="Number of Comments"),
                color=alt.Color('Sentiment', legend=None),
                tooltip=['Sentiment', 'Count']
            ).properties(title="Distribution of Sentiment Categories")
            st.altair_chart(chart, use_container_width=True)

            # Sentiment Scatter Plot
            st.markdown("---")
            st.subheader("🎯 Sentiment Scatter Plot")
            scatter_chart = alt.Chart(sentiment_df).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X('Polarity', title='Polarity (Negative to Positive)'),
                y=alt.Y('Subjectivity', title='Subjectivity (Fact to Opinion)'),
                color=alt.Color('Sentiment', legend=alt.Legend(title="Sentiment")),
                tooltip=['Original Comment', 'Polarity', 'Subjectivity', 'Sentiment']
            ).interactive().properties(title="Polarity vs. Subjectivity of Comments")
            st.altair_chart(scatter_chart, use_container_width=True)

        # LDA Topic Modeling Section
        st.markdown("---")
        st.header("🧠 Topic Modeling (Latent Dirichlet Allocation - LDA)")
        st.write("LDA helps identify underlying topics in your text data.")

        if "sentiment_df" in st.session_state:
            sentiment_df = st.session_state["sentiment_df"]

            processed_texts_for_lda = prepare_gensim_data(sentiment_df["Cleaned Text"].dropna().tolist())

            if not processed_texts_for_lda:
                st.warning("⚠️ No valid text found for topic modeling.")
            else:
                id2word = corpora.Dictionary(processed_texts_for_lda)
                id2word.filter_extremes(no_below=5, no_above=0.5)
                corpus = [id2word.doc2bow(text) for text in processed_texts_for_lda]
                corpus = [doc for doc in corpus if doc]

                if len(corpus) < 3 or len(id2word) < 3:
                    st.warning("⚠️ Not enough documents or vocabulary for LDA.")
                else:
                    max_topics = min(20, len(corpus) - 1, len(id2word))
                    if max_topics < 3:
                        st.warning(f"⚠️ Insufficient data to create at least 3 topics. Max possible: {max_topics}")
                    else:
                        num_topics = st.slider(
                            "Select Number of Topics for LDA",
                            min_value=3,
                            max_value=20,
                            value=min(5, max_topics),
                            step=1
                        )

                        if st.button("🚀 Run LDA Topic Analysis"):
                            with st.spinner(f"Training LDA model with {num_topics} topics..."):
                                lda_model = train_gensim_lda_model(corpus, id2word, num_topics)

                                st.markdown("### 📊 Top Words Per Topic")
                                topic_words = {}
                                for idx, topic in lda_model.print_topics(num_words=10):
                                    st.write(f"**Topic {idx}:** {topic}")
                                    topic_words[idx] = topic

                                # Let user assign topic names manually
                                st.markdown("---")
                                st.subheader("✍️ Assign Names to Topics")
                                topic_names = {}
                                for i in range(num_topics):
                                    default_name = f"Topic {i}"
                                    topic_names[i] = st.text_input(f"Name for Topic {i} (words shown above):", value=default_name, key=f"topic_name_{i}")

                                # Assign numeric topic to each document based on highest topic probability
                                topic_assignments = []
                                for bow in corpus:
                                    topics_per_doc = lda_model.get_document_topics(bow)
                                    top_topic = max(topics_per_doc, key=lambda x: x[1])[0]
                                    topic_assignments.append(top_topic)

                                # Assign topics to sentiment_df but only for rows with non-empty cleaned text
                                non_empty_mask = sentiment_df["Cleaned Text"].str.strip() != ""
                                subset_df = sentiment_df.loc[non_empty_mask].copy()

                                if len(subset_df) == len(topic_assignments):
                                    subset_df["Topic"] = topic_assignments
                                    # Replace numeric topic with user-defined names
                                    subset_df["Topic Name"] = subset_df["Topic"].map(topic_names)
                                else:
                                    st.error("Length mismatch between topic assignments and cleaned text rows!")

                                # Update original dataframe
                                sentiment_df.loc[non_empty_mask, "Topic"] = subset_df["Topic"]
                                sentiment_df.loc[non_empty_mask, "Topic Name"] = subset_df["Topic Name"]

                                # Save updated df back to session state
                                st.session_state["sentiment_df"] = sentiment_df

                                # Show dataframe with assigned topics
                                st.subheader("🗂️ Sentiment Analysis with Assigned Topics")
                                st.dataframe(sentiment_df)

                                # pyLDAvis Visualization
                                st.markdown("---")
                                st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                                st.info("ℹ️ This interactive chart may take a few seconds to load.")
                                vis = gensimvis.prepare(lda_model, corpus, id2word)
                                html_string = pyLDAvis.prepared_data_to_html(vis)
                                st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

        else:
            st.info("☝️ Please run Sentiment Analysis first to generate cleaned comments for topic modeling.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("☝️ Please upload a dataset (CSV, Excel, or TXT) to begin!")
