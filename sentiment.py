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
    logo = Image.open("ucc.png")  # Make sure ucc.png is in your project directory
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

# --- About Section ---
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy.
    This app allows users to analyze the sentiment of comments, generate word clouds, and discover topics using LDA.
    """)

# --- NLTK Resources ---
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

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words)
    return wc.generate(text)

def prepare_gensim_data(texts):
    custom_stopwords = stop_words.union({'from', 'subject', 're', 'edu', 'use'})
    return [
        [word for word in simple_preprocess(str(doc), deacc=True) if word not in custom_stopwords]
        for doc in texts
    ]

@st.cache_resource(show_spinner=False)
def train_gensim_lda_model(corpus, id2word, num_topics):
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=50,
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
            st.error("🚨 Unsupported file format.")
            st.stop()

        text_cols = df.select_dtypes(include="object").columns.tolist()
        selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

        # --- Sentiment Analysis ---
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
            results_df = pd.DataFrame(results)
            st.session_state['results_df'] = results_df  # Store in session_state

            st.subheader("🗂️ Sentiment Analysis Results")
            st.dataframe(results_df)

            st.download_button(
                label="📥 Download Sentiment Results CSV",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

            # --- Word Cloud ---
            st.subheader("☁️ Word Cloud")
            all_cleaned_text = " ".join(results_df["Cleaned Text"].tolist())
            if all_cleaned_text.strip():
                wc_image = generate_wordcloud(all_cleaned_text)
                st.image(wc_image.to_array(), caption="Word Cloud of Cleaned Text", use_container_width=True)
            else:
                st.info("ℹ️ Not enough text for Word Cloud.")

            # --- Sentiment Distribution ---
            st.subheader("📊 Sentiment Distribution")
            counts = results_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y"),
                y=alt.Y('Count'),
                color='Sentiment',
                tooltip=['Sentiment', 'Count']
            ).properties(width=600)
            st.altair_chart(chart)

            # --- Scatter Plot ---
            st.subheader("🎯 Sentiment Scatter Plot")
            scatter_chart = alt.Chart(results_df).mark_circle(size=80).encode(
                x=alt.X('Polarity', title='Polarity'),
                y=alt.Y('Subjectivity', title='Subjectivity'),
                color='Sentiment',
                tooltip=['Original Comment', 'Polarity', 'Subjectivity']
            ).interactive().properties(width=700)
            st.altair_chart(scatter_chart)

        # --- LDA Topic Modeling ---
        st.markdown("---")
        st.header("🧠 Topic Modeling (LDA)")

        if 'results_df' in st.session_state:
            results_df = st.session_state['results_df']
            cleaned_texts = results_df["Cleaned Text"].tolist()
            processed_texts_for_lda = prepare_gensim_data(cleaned_texts)

            id2word = corpora.Dictionary(processed_texts_for_lda)
            id2word.filter_extremes(no_below=5, no_above=0.5)
            corpus = [id2word.doc2bow(text) for text in processed_texts_for_lda]
            corpus = [doc for doc in corpus if doc]  # Filter empty docs

            if len(corpus) < 3 or len(id2word) < 3:
                st.warning("⚠️ Not enough data for LDA.")
            else:
                num_topics = st.slider("Select Number of Topics", 3, 20, 5)

                if st.button("🚀 Run LDA Topic Analysis"):
                    with st.spinner(f"Training LDA model with {num_topics} topics..."):
                        lda_model = train_gensim_lda_model(corpus, id2word, num_topics)

                        # Show top words per topic
                        st.markdown("### 📊 Top Words Per Topic")
                        topics = lda_model.print_topics(num_words=10)
                        for idx, topic in topics:
                            st.write(f"**Topic {idx}:** {topic}")

                        # --- Topic Naming ---
                        topic_names = {}
                        st.markdown("### 📝 Assign Names to Topics")
                        for idx, topic in topics:
                            default_label = f"Topic {idx}"
                            user_input = st.text_input(f"Name for Topic {idx}", value=default_label)
                            topic_names[idx] = user_input

                        # Assign topics to comments
                        dominant_topics = []
                        for bow in corpus:
                            topic_probs = lda_model.get_document_topics(bow)
                            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
                            dominant_topics.append(dominant_topic)

                        topic_labels = [topic_names[t] for t in dominant_topics]

                        # Add to dataframe
                        cleaned_df = results_df.copy()
                        cleaned_df = cleaned_df.iloc[:len(topic_labels)]  # Avoid length mismatch
                        cleaned_df['Assigned Topic'] = topic_labels

                        st.subheader("🗂️ Comments with Assigned Topics")
                        st.dataframe(cleaned_df)

                        st.download_button("📥 Download Topics CSV", cleaned_df.to_csv(index=False), "assigned_topics.csv")

                        # --- LDA Visualization ---
                        st.subheader("📈 Interactive LDA Visualization")
                        vis = gensimvis.prepare(lda_model, corpus, id2word)
                        html_string = pyLDAvis.prepared_data_to_html(vis)
                        st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

        else:
            st.info("Please run Sentiment Analysis first to prepare the cleaned comments.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("☝️ Please upload a dataset to begin analysis.")
