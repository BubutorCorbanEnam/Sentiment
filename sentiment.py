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
from PIL import Image
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# --- Setup ---
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="centered", page_icon="💬")

# --- University Branding ---
col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("ucc.png")  # Ensure ucc.png is in your repo or folder
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment & Topic Analysis App</h4>", unsafe_allow_html=True)

st.markdown("---")

# --- About ---
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Developed by Bubutor Corban Enam during an NLP training session by Professor Andy.  
    - Sentiment Analysis  
    - Word Cloud  
    - LDA Topic Modeling (Manual Topic Naming)  
    """)

# --- NLTK Setup ---
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

# --- Upload Section ---
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
        selected_col = st.selectbox("Select Text Column", text_cols)

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

            st.subheader("🗂️ Sentiment Analysis Results")
            st.dataframe(results_df)

            st.download_button("📥 Download CSV", results_df.to_csv(index=False), file_name="sentiment_results.csv")

            # --- Word Cloud ---
            st.subheader("☁️ Word Cloud")
            all_text = " ".join(results_df["Cleaned Text"].tolist())
            if len(all_text.strip()) > 0:
                wc_image = generate_wordcloud(all_text)
                st.image(wc_image.to_array(), caption="Word Cloud", use_container_width=True)

            # --- Sentiment Distribution ---
            st.subheader("📊 Sentiment Distribution")
            counts = results_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y"),
                y='Count',
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
        st.subheader("🧠 Topic Modeling (LDA)")

        processed_texts = prepare_gensim_data(df[selected_col].dropna().tolist())
        id2word = corpora.Dictionary(processed_texts)
        id2word.filter_extremes(no_below=3, no_above=0.5)
        corpus = [id2word.doc2bow(text) for text in processed_texts]
        corpus = [doc for doc in corpus if doc]

        if len(corpus) < 3 or len(id2word) < 3:
            st.warning("Not enough data for LDA after filtering.")
        else:
            max_topics = min(len(corpus)-1, len(id2word))
            num_topics = st.slider("Select Number of Topics", 3, max_topics if max_topics>3 else 3, min(5, max_topics) if max_topics>=5 else 3)

            if st.button("🚀 Run LDA Analysis"):
                lda_model = train_gensim_lda_model(corpus, id2word, num_topics)

                st.markdown("### 🗒️ Name Your Topics")
                topic_labels = {}
                for idx, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
                    words = ", ".join([w for w, _ in topic])
                    label = st.text_input(f"Label for Topic {idx+1} (Top Words: {words})", value=f"Topic {idx+1}")
                    topic_labels[idx] = label

                # --- Document-Level Topic Assignment ---
                st.subheader("📄 Document Topic Assignment")
                dominant_topics = []
                for doc_bow in corpus:
                    topic_probs = lda_model.get_document_topics(doc_bow)
                    dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
                    dominant_topics.append(topic_labels[dominant_topic])

                df["Assigned Topic"] = dominant_topics[:len(df)]
                st.dataframe(df[[selected_col, "Assigned Topic"]])

                st.download_button(
                    "📥 Download LDA Topics CSV",
                    df.to_csv(index=False),
                    file_name="lda_topics.csv"
                )

                # --- pyLDAvis ---
                st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                with st.spinner("Generating visualization..."):
                    vis = gensimvis.prepare(lda_model, corpus, id2word)
                    html_string = pyLDAvis.prepared_data_to_html(vis)
                    st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("☝️ Upload your data to begin.")
