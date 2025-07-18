import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import altair as alt
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from PIL import Image

# --- Setup ---
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="centered", page_icon="💬")

# --- Branding ---
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
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy. This app allows users to analyze sentiments, generate word clouds, and discover topics using LDA.   
    """)

# --- NLTK Setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# --- Streamlit App ---
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

        if "sentiment_df" not in st.session_state:
            if st.button("🔍 Run Sentiment Analysis & Word Cloud"):
                results = []
                for comment in df[selected_col].dropna():
                    cleaned = clean_text(comment)
                    polarity, subjectivity, sentiment, opinion = analyze_sentiment(cleaned)
                    results.append({
                        "Original": comment,
                        "Cleaned": cleaned,
                        "Polarity": polarity,
                        "Subjectivity": subjectivity,
                        "Sentiment": sentiment,
                        "Type": opinion
                    })
                sentiment_df = pd.DataFrame(results)
                st.session_state["sentiment_df"] = sentiment_df

        if "sentiment_df" in st.session_state:
            sentiment_df = st.session_state["sentiment_df"]

            st.subheader("🗂️ Sentiment Analysis Results")
            st.dataframe(sentiment_df)

            st.download_button("📥 Download Sentiment Results", sentiment_df.to_csv(index=False), "sentiment_results.csv")

            # --- Word Cloud ---
            st.subheader("☁️ Word Cloud")
            wc_text = " ".join(sentiment_df["Cleaned"].tolist())
            wc_image = generate_wordcloud(wc_text)
            st.image(wc_image.to_array(), use_container_width=True)

            # --- Sentiment Bar Plot ---
            st.subheader("📊 Sentiment Distribution")
            counts = sentiment_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            bar_chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y"),
                y='Count',
                color='Sentiment',
                tooltip=['Sentiment', 'Count']
            ).properties(width=600)
            st.altair_chart(bar_chart)

            # --- Scatter Plot ---
            st.subheader("🎯 Sentiment Scatter Plot")
            scatter_chart = alt.Chart(sentiment_df).mark_circle(size=80).encode(
                x=alt.X('Polarity', title='Polarity'),
                y=alt.Y('Subjectivity', title='Subjectivity'),
                color='Sentiment',
                tooltip=['Original', 'Polarity', 'Subjectivity']
            ).interactive().properties(width=700)
            st.altair_chart(scatter_chart)

            # --- LDA Topic Modeling ---
            st.header("🧠 Topic Modeling (LDA)")

            if "lda_model" not in st.session_state:
                processed_texts = prepare_gensim_data(sentiment_df["Cleaned"])
                id2word = corpora.Dictionary(processed_texts)
                id2word.filter_extremes(no_below=5, no_above=0.5)
                corpus = [id2word.doc2bow(text) for text in processed_texts]

                corpus = [doc for doc in corpus if doc]

                num_topics = st.slider("Select Number of Topics", 3, 20, 5)

                lda_model = gensim.models.LdaModel(
                    corpus=corpus,
                    id2word=id2word,
                    num_topics=num_topics,
                    random_state=50,
                    per_word_topics=True
                )

                st.session_state["lda_model"] = lda_model
                st.session_state["corpus"] = corpus
                st.session_state["id2word"] = id2word
                st.session_state["num_topics"] = num_topics

            lda_model = st.session_state["lda_model"]
            corpus = st.session_state["corpus"]
            id2word = st.session_state["id2word"]
            num_topics = st.session_state["num_topics"]

            st.subheader("🔤 LDA Topics - Top Words")
            topic_words = {}
            for idx, topic in lda_model.show_topics(formatted=False, num_words=10):
                words = ", ".join([w for w, _ in topic])
                topic_words[idx] = words
                st.write(f"**Topic {idx+1}:** {words}")

            # --- Manual Topic Assignment ---
            if "topic_labels" not in st.session_state:
                st.session_state["topic_labels"] = {i: f"Topic {i+1}" for i in range(num_topics)}

            st.markdown("### 📝 Assign Custom Labels to Topics")
            for i in range(num_topics):
                default_label = st.session_state["topic_labels"].get(i, f"Topic {i+1}")
                label = st.text_input(f"Label for Topic {i+1} ({topic_words[i]})", value=default_label, key=f"label_{i}")
                st.session_state["topic_labels"][i] = label

            # --- Predict Topic for Each Comment ---
            topic_assignments = []
            processed_texts = prepare_gensim_data(sentiment_df["Cleaned"])
            for bow in [id2word.doc2bow(text) for text in processed_texts]:
                if bow:
                    topic_prob = lda_model.get_document_topics(bow)
                    top_topic = max(topic_prob, key=lambda x: x[1])[0]
                    topic_assignments.append(top_topic)
                else:
                    topic_assignments.append(-1)

            sentiment_df["Topic"] = [st.session_state["topic_labels"].get(t, "Unassigned") if t != -1 else "Unassigned" for t in topic_assignments]

            st.subheader("📄 Comments with Assigned Topics")
            st.dataframe(sentiment_df[["Original", "Cleaned", "Topic"]])

            # --- pyLDAvis with Custom Labels ---
            st.subheader("📈 Interactive LDA Visualization")
            vis = gensimvis.prepare(lda_model, corpus, id2word)

            # Replace pyLDAvis topic labels
            for i, row in vis.topic_info.iterrows():
                topic_id = int(row['Category'].replace('Topic', '').strip()) - 1
                if topic_id in st.session_state["topic_labels"]:
                    vis.topic_info.at[i, 'Category'] = st.session_state["topic_labels"][topic_id]

            html_string = pyLDAvis.prepared_data_to_html(vis)
            st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
