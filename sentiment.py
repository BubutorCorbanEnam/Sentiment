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
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Setup
st.set_page_config(page_title="UCC Sentiment & Topic App", layout="centered", page_icon="💬")
col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("ucc.png")
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment & Topic Modeling App</h4>", unsafe_allow_html=True)

st.markdown("---")

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
def train_gensim_lda_model(corpus, _id2word, num_topics):
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=_id2word,
        num_topics=num_topics,
        random_state=50,
        per_word_topics=True
    )
    return lda_model

# App logic
uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or TXT", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file:
    if "sentiment_done" not in st.session_state:
        st.session_state["sentiment_done"] = False
    if "sentiment_df" not in st.session_state:
        st.session_state["sentiment_df"] = None

    df = None
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment"])

    text_cols = df.select_dtypes(include="object").columns.tolist()
    selected_col = st.selectbox("Select Text Column", text_cols)

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
        st.session_state["sentiment_done"] = True

        st.subheader("🗂️ Sentiment Analysis Results")
        st.dataframe(sentiment_df)

        st.download_button(
            label="📥 Download Sentiment CSV",
            data=sentiment_df.to_csv(index=False).encode('utf-8'),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )

        st.subheader("☁️ Word Cloud")
        all_cleaned_text = " ".join(sentiment_df["Cleaned Text"].tolist())
        wc_image = generate_wordcloud(all_cleaned_text)
        st.image(wc_image.to_array(), use_container_width=True)

        st.subheader("📊 Sentiment Distribution")
        counts = sentiment_df['Sentiment'].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X('Sentiment', sort="-y"),
            y='Count',
            color='Sentiment',
            tooltip=['Sentiment', 'Count']
        ).properties(width=600)
        st.altair_chart(chart)

        st.subheader("🎯 Sentiment Scatter Plot")
        scatter_chart = alt.Chart(sentiment_df).mark_circle(size=80).encode(
            x=alt.X('Polarity', title='Polarity'),
            y=alt.Y('Subjectivity', title='Subjectivity'),
            color='Sentiment',
            tooltip=['Original Comment', 'Polarity', 'Subjectivity']
        ).interactive().properties(width=700)
        st.altair_chart(scatter_chart)

    if st.session_state["sentiment_done"]:
        st.markdown("---")
        st.header("🧠 Topic Modeling (LDA)")

        clean_comments = st.session_state["sentiment_df"]["Cleaned Text"].tolist()
        processed_texts = prepare_gensim_data(clean_comments)
        _id2word = corpora.Dictionary(processed_texts)
        _id2word.filter_extremes(no_below=5, no_above=0.5)
        corpus = [_id2word.doc2bow(text) for text in processed_texts]
        corpus = [doc for doc in corpus if doc]

        if len(corpus) >= 3 and len(_id2word) >= 3:
            num_topics = st.slider("Select Number of Topics", 3, 20, 5)
            if st.button("🚀 Run LDA Analysis"):
                lda_model = train_gensim_lda_model(corpus, _id2word, num_topics)

                st.subheader("🔑 LDA Topics - Top Words")
                topics = lda_model.print_topics(num_words=10)
                for idx, topic in topics:
                    st.write(f"**Topic {idx+1}:** {topic}")

                topic_names = {}
                st.subheader("✍️ Assign Descriptive Names to Topics")
                for idx, _ in enumerate(topics):
                    topic_names[idx] = st.text_input(f"Topic {idx+1} Name", key=f"topic_name_{idx}")

                if st.button("✅ Finalize Topic Assignment"):
                    topic_assignments = []
                    for doc in corpus:
                        topic_dist = lda_model.get_document_topics(doc)
                        if topic_dist:
                            top_topic = max(topic_dist, key=lambda x: x[1])[0]
                            topic_assignments.append(topic_names.get(top_topic, f"Topic {top_topic+1}"))
                        else:
                            topic_assignments.append("Unassigned")

                    st.session_state["sentiment_df"]["Topic"] = topic_assignments
                    st.subheader("📝 Final Labeled Topics")
                    st.dataframe(st.session_state["sentiment_df"])

                    st.download_button(
                        label="📥 Download LDA Topics CSV",
                        data=st.session_state["sentiment_df"].to_csv(index=False).encode('utf-8'),
                        file_name="lda_topics_assigned.csv",
                        mime="text/csv"
                    )

                st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                with st.spinner("Generating visualization..."):
                    vis = gensimvis.prepare(lda_model, corpus, _id2word)
                    html_string = pyLDAvis.prepared_data_to_html(vis)
                    st.components.v1.html(html_string, width=1000, height=800, scrolling=True)
        else:
            st.warning("Not enough unique tokens or documents for LDA after preprocessing.")

else:
    st.info("☝️ Please upload a dataset (CSV, Excel, or TXT) to begin your analysis.")
