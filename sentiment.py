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

        text_cols = df.select_dtypes(include="object").columns.tolist()
        selected_col = st.selectbox("Select Text Column", text_cols)

        if st.button("🔍 Run Sentiment Analysis & WordCloud"):
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

            st.download_button("📥 Download CSV", sentiment_df.to_csv(index=False), "sentiment_results.csv")

            st.subheader("☁️ Word Cloud")
            all_text = " ".join(sentiment_df["Cleaned"].tolist())
            wc_image = generate_wordcloud(all_text)
            st.image(wc_image.to_array())

            st.subheader("📊 Sentiment Distribution")
            counts = sentiment_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y"),
                y='Count',
                color='Sentiment',
                tooltip=['Sentiment', 'Count']
            )
            st.altair_chart(chart)

            st.subheader("🎯 Scatter Plot (Polarity vs Subjectivity)")
            scatter = alt.Chart(sentiment_df).mark_circle(size=80).encode(
                x='Polarity', y='Subjectivity', color='Sentiment',
                tooltip=['Original', 'Polarity', 'Subjectivity']
            ).interactive()
            st.altair_chart(scatter)

            st.subheader("📊 Subjectivity Sentiment Analysis")
            plt.figure(figsize=(15,10))
            plt.title('Subjectivity Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Counts')
            sentiment_df['Type'].value_counts().plot(kind='bar', color=sns.color_palette('rocket'))
            st.pyplot(plt.gcf())
            plt.clf()

    except Exception as e:
        st.error(f"Error: {e}")

# --- LDA Topic Modeling ---
if st.session_state["sentiment_df"] is not None:
    st.markdown("---")
    st.header("🧠 LDA Topic Modeling (Manual Topic Assignment Supported)")

    clean_comments = st.session_state["sentiment_df"]["Cleaned"].dropna().tolist()
    processed_texts = prepare_gensim_data(clean_comments)
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
        for idx, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
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
            topic_assignments = []
            for bow in corpus:
                topic_probs = st.session_state["lda_model"].get_document_topics(bow)
                if topic_probs:
                    assigned_topic = max(topic_probs, key=lambda x: x[1])[0]
                    label = st.session_state["topic_labels"].get(assigned_topic, f"Topic {assigned_topic+1}")
                    topic_assignments.append(label)
                else:
                    topic_assignments.append("Unassigned")

            st.session_state["sentiment_df"]["Topic"] = topic_assignments
            df_1 = st.session_state["sentiment_df"].copy()

            st.success("Custom topics assigned successfully.")
            st.dataframe(df_1)

            st.subheader("📊 Topic Analysis Based on Manual Labels")
            plt.figure(figsize=(15,10))
            plt.title('Topic Analysis')
            plt.xlabel('Topic')
            plt.ylabel('Counts')
            sns.barplot(x=df_1['Topic'].value_counts().index, y=df_1['Topic'].value_counts().values, palette=sns.color_palette('flare'))
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("📊 Topic Polarity Distribution")
            df_topic_polarity = df_1.groupby('Topic')['Sentiment'].value_counts().unstack(fill_value=0).apply(lambda x: x / x.sum() * 100, axis=1)
            polarity_map = {"😠 Negative": "Negative", "😐 Neutral": "Neutral", "😊 Positive": "Positive"}
            df_topic_polarity.rename(columns=polarity_map, inplace=True)
            colors = ["red", "yellow", "green"]
            ax = df_topic_polarity.plot(kind='bar', color=colors, stacked=True, figsize=(15, 10))
            ax.set_xlabel('Topic')
            ax.set_ylabel('% Polarity')
            ax.set_title('Topic Polarity Distribution')
            ax.set_xticklabels(df_topic_polarity.index, rotation=90)
            ax.legend(title='Polarity')
            st.pyplot(ax.get_figure())
            plt.clf()

            # --- NetworkX Graph based on actual topic polarity and manual topic labels ---
            
            st.subheader("🔗 Topic Relationship Graph (Manual Topic Names)")
            
            # Use actual assigned topics in the dataframe (manual labels from assignment step)
            df_1 = st.session_state["sentiment_df"].copy()
            
            # Ensure consistent sentiment columns for similarity calculation
            expected_sentiments = ["😠 Negative", "😐 Neutral", "😊 Positive"]
            
            # Compute sentiment proportions per topic
            df_topic_sentiment = df_1.groupby('Topic')['Sentiment'].value_counts(normalize=True).unstack().fillna(0)
            
            # Ensure all expected sentiment columns exist
            for col in expected_sentiments:
                if col not in df_topic_sentiment.columns:
                    df_topic_sentiment[col] = 0.0
            
            # Reorder columns to match expected sentiments
            df_topic_sentiment = df_topic_sentiment[expected_sentiments]
            
            # Compute cosine similarity matrix between topics
            topic_polarity_matrix = cosine_similarity(df_topic_sentiment.values)
            
            # Use manually assigned topic names as nodes
            topic_names = df_topic_sentiment.index.tolist()
            
            # Zero diagonal to remove self-loops
            np.fill_diagonal(topic_polarity_matrix, 0)
            
            # Create graph
            G = nx.Graph()
            G.add_nodes_from(topic_names)
            
            # Add edges where similarity > threshold
            threshold = 0.5
            for i in range(len(topic_names)):
                for j in range(i+1, len(topic_names)):
                    weight = topic_polarity_matrix[i][j]
                    if weight > threshold:
                        G.add_edge(topic_names[i], topic_names[j], weight=weight)
            
            # Draw the graph
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            
            nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=1500, edge_color='gray')
            
            # Draw edge labels with weights
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
