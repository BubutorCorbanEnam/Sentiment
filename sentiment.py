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
    logo = Image.open("ucc.png")  # Make sure ucc.png is in the project directory
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

        # CLEAN ONLY NON-NULL TEXTS
        non_null_texts = df[selected_col].dropna().copy()
        cleaned_texts = non_null_texts.apply(clean_text)

        # --- Sentiment Analysis and Word Cloud on CLEANED texts ---
        if st.button("🔍 Run Sentiment Analysis & Word Cloud"):
            with st.spinner("Analyzing sentiment and generating word cloud..."):
                results = []
                for orig_text, clean_text_ in zip(non_null_texts, cleaned_texts):
                    polarity, subjectivity, sentiment, opinion = analyze_sentiment(clean_text_)
                    results.append({
                        "Original Comment": orig_text,
                        "Cleaned Text": clean_text_,
                        "Polarity": polarity,
                        "Subjectivity": subjectivity,
                        "Sentiment": sentiment,
                        "Type": opinion
                    })

                results_df = pd.DataFrame(results)

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

        # --- LDA Topic Modeling on CLEANED texts ---
        st.markdown("---")
        st.header("🧠 Topic Modeling (Latent Dirichlet Allocation - LDA)")
        st.write("LDA helps identify underlying topics in your text data.")

        processed_texts_for_lda = prepare_gensim_data(cleaned_texts.tolist())

        if not processed_texts_for_lda:
            st.warning("⚠️ No valid text found for topic modeling after cleaning. Please check your data.")
        else:
            id2word = corpora.Dictionary(processed_texts_for_lda)
            id2word.filter_extremes(no_below=5, no_above=0.5)
            corpus = [id2word.doc2bow(text) for text in processed_texts_for_lda]
            corpus = [doc for doc in corpus if doc]

            if len(corpus) < 3 or len(id2word) < 3:
                st.warning("⚠️ Not enough unique documents or vocabulary for LDA topic modeling. Need at least 3 documents and 3 unique words after processing.")
            else:
                max_topics = min(20, len(corpus) - 1, len(id2word))
                if max_topics < 3:
                    st.warning(f"⚠️ Insufficient data to create at least 3 topics after filtering. Only {max_topics} topics can be generated.")
                else:
                    num_topics = st.slider(
                        "Select Number of Topics for LDA (More topics can be harder to interpret)",
                        min_value=3,
                        max_value=max_topics,
                        value=min(5, max_topics),
                        step=1
                    )

                    if st.button("🚀 Run LDA Topic Analysis"):
                        with st.spinner(f"Training LDA model with {num_topics} topics... This might take a moment."):
                            try:
                                lda_model = train_gensim_lda_model(corpus, id2word, num_topics)

                                st.markdown("### 📊 Top Words Per Topic")
                                lda_topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
                                for topic_num, word_probs in lda_topics:
                                    words_str = " + ".join([f'{round(prob,3)}*"{word}"' for word, prob in word_probs])
                                    st.write(f"**Topic {topic_num}:** {words_str}")

                                st.subheader("✏️ Assign Meaningful Labels to Each Topic")
                                labels = []
                                for i in range(num_topics):
                                    label = st.text_input(f"Label for Topic {i}", key=f"topic_label_{i}")
                                    labels.append(label)

                                if st.button("🗂️ Assign Topics to Comments"):
                                    topic_assignments = []
                                    for bow in corpus:
                                        doc_topics = lda_model.get_document_topics(bow)
                                        dominant_topic = max(doc_topics, key=lambda x: x[1])[0]
                                        topic_assignments.append(dominant_topic)

                                    # Create DataFrame for the cleaned subset with topic assignments
                                    lda_results_df = pd.DataFrame({
                                        selected_col: non_null_texts.values,
                                        "Cleaned Text": cleaned_texts.values,
                                        "Topic Number": topic_assignments
                                    })

                                    # Assign user labels if provided, else fallback topic number
                                    lda_results_df["Topic Label"] = lda_results_df["Topic Number"].apply(
                                        lambda x: labels[x].strip() if labels[x].strip() else f"Topic {x}"
                                    )

                                    st.subheader("📊 Comments with Assigned Topics")
                                    st.dataframe(lda_results_df)

                                    csv = lda_results_df.to_csv(index=False).encode('utf-8')
                                    st.download_button("📥 Download Comments with Topics", data=csv, file_name="comments_with_topics.csv", mime="text/csv")

                                st.markdown("---")
                                st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                                with st.spinner("Generating interactive visualization..."):
                                    vis = gensimvis.prepare(lda_model, corpus, id2word)
                                    html_string = pyLDAvis.prepared_data_to_html(vis)
                                    st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

                            except Exception as e:
                                st.error(f"An error occurred during LDA topic modeling: {e}")
                                st.warning("Try reducing the number of topics or ensuring diverse data.")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing your file: {e}")
        st.info("Please ensure your file is correctly formatted and the selected column contains text data.")

else:
    st.info("☝️ Please upload a dataset (CSV, Excel, or TXT) to begin your text analysis journey!")
