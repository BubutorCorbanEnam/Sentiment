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
    logo = Image.open("ucc.png")  # Make sure ucc.png is in your directory
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

# Download NLTK data
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
def train_gensim_lda_model(corpus, id2word, num_topics):
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
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
        if not text_cols:
            st.warning("No text columns found.")
            st.stop()

        selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

        # Run Sentiment Analysis
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
            st.session_state["sentiment_df"] = sentiment_df  # Save to session state

            # Show Sentiment Results
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
            st.subheader("☁️ Word Cloud")
            all_text = " ".join(sentiment_df["Cleaned Text"].tolist())
            if all_text.strip():
                wc_image = generate_wordcloud(all_text)
                st.image(wc_image.to_array(), use_container_width=True)
            else:
                st.info("Not enough text for Word Cloud.")

            # Sentiment Distribution Bar Plot
            st.subheader("📊 Sentiment Distribution")
            counts = sentiment_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            bar_chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort='-y'),
                y='Count',
                color='Sentiment',
                tooltip=['Sentiment', 'Count']
            ).properties(width=600)
            st.altair_chart(bar_chart)

            # Sentiment Scatter Plot
            st.subheader("🎯 Sentiment Scatter Plot")
            scatter = alt.Chart(sentiment_df).mark_circle(size=80).encode(
                x='Polarity',
                y='Subjectivity',
                color='Sentiment',
                tooltip=['Original Comment', 'Polarity', 'Subjectivity']
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)

        # LDA Topic Modeling (after sentiment is run)
        if "sentiment_df" in st.session_state:
            sentiment_df = st.session_state["sentiment_df"]

            st.markdown("---")
            st.header("🧠 Topic Modeling (Latent Dirichlet Allocation - LDA)")

            # Prepare data for LDA
            processed_texts_for_lda = prepare_gensim_data(sentiment_df["Cleaned Text"].dropna().tolist())
            id2word = corpora.Dictionary(processed_texts_for_lda)
            id2word.filter_extremes(no_below=5, no_above=0.5)
            corpus = [id2word.doc2bow(text) for text in processed_texts_for_lda]
            corpus = [doc for doc in corpus if doc]

            if len(corpus) < 3 or len(id2word) < 3:
                st.warning("Not enough data for LDA topic modeling (need at least 3 documents and 3 unique words).")
            else:
                max_topics = min(len(corpus) - 1, len(id2word), 20)
                num_topics = st.slider(
                    "Select Number of Topics for LDA (max 20)",
                    min_value=3,
                    max_value=20,
                    value=min(5, max_topics)
                )

                if st.button("🚀 Run LDA Topic Analysis"):
                    lda_model = train_gensim_lda_model(corpus, id2word, num_topics)
                    st.session_state["lda_model"] = lda_model
                    st.session_state["corpus"] = corpus
                    st.session_state["id2word"] = id2word

                    # Show topics and top words
                    st.markdown("### 📊 Top Words Per Topic")
                    topic_words = {}
                    for idx, topic in lda_model.print_topics(num_words=10):
                        st.write(f"**Topic {idx}:** {topic}")
                        topic_words[idx] = topic

                    # Assign dominant topic per document
                    topic_assignments = []
                    for bow in corpus:
                        topics_per_doc = lda_model.get_document_topics(bow)
                        top_topic = max(topics_per_doc, key=lambda x: x[1])[0]
                        topic_assignments.append(top_topic)

                    # Add Topic column to DataFrame subset
                    non_empty_mask = sentiment_df["Cleaned Text"].str.strip() != ""
                    subset_df = sentiment_df.loc[non_empty_mask].copy()
                    subset_df["Topic"] = topic_assignments
                    sentiment_df.loc[non_empty_mask, "Topic"] = topic_assignments
                    st.session_state["sentiment_df"] = sentiment_df
                    st.session_state["topic_assignments"] = topic_assignments

            # Topic renaming and mapping (only if LDA model and assignments exist)
            if ("topic_assignments" in st.session_state) and ("sentiment_df" in st.session_state):
                sentiment_df = st.session_state["sentiment_df"]
                unique_topics = list(set(st.session_state["topic_assignments"]))
                unique_topics.sort()
                st.subheader("✍️ Assign Names to Topics")

                if "topic_names" not in st.session_state:
                    # Initialize with default names
                    st.session_state["topic_names"] = {i: f"Topic {i}" for i in unique_topics}

                # Inputs for user to assign topic names
                for i in unique_topics:
                    st.session_state["topic_names"][i] = st.text_input(
                        f"Name for Topic {i}",
                        value=st.session_state["topic_names"][i],
                        key=f"topic_name_{i}"
                    )

                # Map numeric topic to user assigned names
                sentiment_df["Topic Name"] = sentiment_df["Topic"].map(st.session_state["topic_names"])
                st.session_state["sentiment_df"] = sentiment_df

                st.subheader("🗂️ Sentiment Results with Assigned Topic Names")
                st.dataframe(sentiment_df)

                # Show pyLDAvis interactive visualization
                if "lda_model" in st.session_state and "corpus" in st.session_state and "id2word" in st.session_state:
                    st.markdown("---")
                    st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                    with st.spinner("Generating visualization..."):
                        vis = gensimvis.prepare(
                            st.session_state["lda_model"],
                            st.session_state["corpus"],
                            st.session_state["id2word"],
                        )
                        html_string = pyLDAvis.prepared_data_to_html(vis)
                        st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("☝️ Please upload a dataset (CSV, Excel, or TXT) to begin your text analysis journey!")
