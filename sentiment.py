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
import numpy as np
from gensim import corpora
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

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
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy.
    This app allows users to analyze sentiment and discover topics in text data via LDA.
    """)

# --- NLTK Setup ---
nltk.download('punkt_tab')
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
    if polarity > 0:
        sentiment = "😊 Positive"
    elif polarity < 0:
        sentiment = "😠 Negative"
    else:
        sentiment = "😐 Neutral"
    opinion = "Opinion" if subjectivity > 0 else "Fact"
    return polarity, subjectivity, sentiment, opinion

@st.cache_resource(show_spinner=False)
def train_gensim_lda_model(_corpus, _id2word, _num_topics):
    lda_model = gensim.models.LdaModel(
        corpus=_corpus,
        id2word=_id2word,
        num_topics=_num_topics,
        random_state=50,
        passes=5,
        iterations=50,
        per_word_topics=True
    )
    return lda_model

# --- Session State ---
if "sentiment_df" not in st.session_state:
    st.session_state["sentiment_df"] = None
if "topic_labels" not in st.session_state:
    st.session_state["topic_labels"] = {}
if "lda_model" not in st.session_state:
    st.session_state["lda_model"] = None
if "corpus" not in st.session_state:
    st.session_state["corpus"] = None
if "id2word" not in st.session_state:
    st.session_state["id2word"] = None
if "num_topics" not in st.session_state:
    st.session_state["num_topics"] = 5

# --- Upload ---
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

        if st.button("🔍 Run Sentiment Analysis"):
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

            st.subheader("🗂️ Sentiment Analysis Results")
            st.dataframe(sentiment_df)

    except Exception as e:
        st.error(f"Error: {e}")

# --- LDA Topic Modeling ---
if st.session_state["sentiment_df"] is not None:
    st.markdown("---")
    st.header("🧠 LDA Topic Modeling (Manual Topic Assignment Supported)")

    clean_comments = st.session_state["sentiment_df"]["Cleaned"].dropna().tolist()
    processed_texts = [simple_preprocess(doc) for doc in clean_comments]
    id2word = corpora.Dictionary(processed_texts)
    corpus = [id2word.doc2bow(text) for text in processed_texts]

    st.session_state["id2word"] = id2word
    st.session_state["corpus"] = corpus

    num_topics = st.slider("Select Number of Topics", 3, 20, 5)
    st.session_state["num_topics"] = num_topics

    if st.button("🚀 Run LDA"):
        lda_model = train_gensim_lda_model(corpus, id2word, num_topics)
        st.session_state["lda_model"] = lda_model

        st.subheader("🔑 Top Words per Topic")
        for idx, topic in lda_model.show_topics(num_topics=num_topics, num_words=30, formatted=False):
            words = ", ".join([w for w, p in topic])
            st.write(f"**Topic {idx+1}:** {words}")

    if st.session_state["lda_model"]:
        st.markdown("---")
        st.subheader("📝 Assign Custom Labels to Topics")

        with st.form("topic_label_form"):
            for i in range(num_topics):
                default_label = st.session_state["topic_labels"].get(i, f"Topic {i+1}")
                new_label = st.text_input(f"Label for Topic {i+1}", value=default_label, key=f"topic_input_{i}")
                st.session_state["topic_labels"][i] = new_label
            submit_labels = st.form_submit_button("✔️ Apply Topic Labels")

        if submit_labels:
            df_1 = st.session_state["sentiment_df"].copy()
            topic_assignments = []
            for bow in corpus:
                topic_probs = st.session_state["lda_model"].get_document_topics(bow)
                if topic_probs:
                    assigned_topic = max(topic_probs, key=lambda x: x[1])[0]
                    label = st.session_state["topic_labels"].get(assigned_topic, f"Topic {assigned_topic+1}")
                    topic_assignments.append(label)
                else:
                    topic_assignments.append("Unassigned")

            df_1["Topic"] = topic_assignments

            st.subheader("📊 Topic Analysis Based on Manual Labels")
            plt.figure(figsize=(15,10))
            sns.countplot(data=df_1, x="Topic", palette="flare")
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("📊 Topic Polarity Distribution")
            df_topic_polarity = df_1.groupby('Topic')['Sentiment'].value_counts().unstack(fill_value=0).apply(lambda x: x / x.sum() * 100, axis=1)
            polarity_map = {"😠 Negative": "Negative", "😐 Neutral": "Neutral", "😊 Positive": "Positive"}
            df_topic_polarity.rename(columns=polarity_map, inplace=True)

            color_mapping = {'Negative': 'red', 'Neutral': 'yellow', 'Positive': 'green'}
            colors = [color_mapping.get(col, 'gray') for col in df_topic_polarity.columns]

            ax = df_topic_polarity.plot(kind='bar', color=colors, stacked=True, figsize=(15, 10))
            ax.set_xlabel('Topic')
            ax.set_ylabel('% Polarity')
            ax.set_title('Topic Polarity Distribution')
            ax.set_xticklabels(df_topic_polarity.index, rotation=90)
            ax.legend(title='Polarity')

            st.pyplot(ax.get_figure())
            plt.clf()

            st.subheader("🔗 Topic Relationship Graph (Manual Topic Names)")
            df_topic_sentiment = df_1.groupby('Topic')['Sentiment'].value_counts(normalize=True).unstack().fillna(0)
            for col in ["😠 Negative", "😐 Neutral", "😊 Positive"]:
                if col not in df_topic_sentiment.columns:
                    df_topic_sentiment[col] = 0
            df_topic_sentiment = df_topic_sentiment[["😠 Negative", "😐 Neutral", "😊 Positive"]]

            topic_names = df_topic_sentiment.index.tolist()
            topic_polarity_matrix = cosine_similarity(df_topic_sentiment.values)
            np.fill_diagonal(topic_polarity_matrix, 0)

            G = nx.Graph()
            G.add_nodes_from(topic_names)

            for i in range(len(topic_polarity_matrix)):
                for j in range(len(topic_polarity_matrix[0])):
                    if topic_polarity_matrix[i][j] > 0.5:
                        G.add_edge(topic_names[i], topic_names[j], weight=topic_polarity_matrix[i][j])

            pos = nx.spring_layout(G)
            plt.figure(figsize=(12,8))
            nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=1500, edge_color='gray')
            edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("📈 Interactive LDA Visualization")
            vis = gensimvis.prepare(st.session_state["lda_model"], corpus, id2word)
            html_string = pyLDAvis.prepared_data_to_html(vis)
            st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

else:
    st.info("Upload data and run sentiment analysis first to enable topic modeling.")
