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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PIL import Image
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Page configuration
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="wide", page_icon="💬")

# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Setup ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Custom CSS ---
st.markdown("""
    <style>
        .main { background-color: #f4f6f9; }
        .stButton>button { background-color: #002147; color: white; border-radius: 8px; }
        .stDownloadButton>button { background-color: #FFD700; color: black; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Password ---
PASSWORD = "CORBAN"
user_password = st.text_input("🔒 Enter Password:", type="password")
if user_password != PASSWORD:
    st.warning("Enter the correct password to access the app.")
    st.stop()

st.title("University of Cape Coast - Sentiment & Topic Analysis Portal")

# --- Session State ---
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Original", "Cleaned", "Polarity", "Subjectivity", "Sentiment", "Type"])

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
    opinion_type = "Opinion" if subjectivity > 0 else "Fact"
    return polarity, subjectivity, sentiment, opinion_type

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words)
    return wc.generate(text)

def prepare_gensim_lda(texts, num_topics=5):
    tokenized = [simple_preprocess(doc) for doc in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    return lda_model, dictionary, corpus

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
        selected_col = st.selectbox("Select Text Column", text_cols)

        if st.button("🔍 Analyze"):
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
            results_df = pd.DataFrame(results)
            st.session_state.results_df = results_df

            st.subheader("🗂️ Analysis Results")
            st.dataframe(results_df)

            st.download_button("📥 Download Results", data=results_df.to_csv(index=False), file_name="sentiment_results.csv")

            # --- WordCloud ---
            st.subheader("☁️ Word Cloud")
            all_text = " ".join(results_df["Cleaned"].tolist())
            if len(all_text.strip()) > 0:
                wc_image = generate_wordcloud(all_text)
                st.image(wc_image.to_array(), caption="Word Cloud", use_column_width=True)
            else:
                st.warning("Not enough text for WordCloud. Please check your input.")

            # --- Sentiment Distribution ---
            st.subheader("📊 Sentiment Distribution")
            counts = results_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x='Sentiment',
                y='Count',
                color='Sentiment',
                tooltip=['Sentiment', 'Count']
            ).properties(width=600)
            st.altair_chart(chart)

            # --- Scatter Plot of Polarity vs Subjectivity ---
            st.subheader("📈 Sentiment Scatter Plot")
            scatter = alt.Chart(results_df).mark_circle(size=60).encode(
                x='Polarity',
                y='Subjectivity',
                color='Sentiment',
                tooltip=['Original', 'Polarity', 'Subjectivity', 'Sentiment']
            ).interactive().properties(width=700)
            st.altair_chart(scatter)

            # --- LDA ---
            st.subheader("🧠 Topic Modeling (LDA)")
            num_topics = st.slider("Number of Topics", 3, 15, 5)
            lda_model, dictionary, corpus = prepare_gensim_lda(results_df["Cleaned"].tolist(), num_topics)

            st.markdown("**📖 LDA Dictionary (token2id):**")
            dict_df = pd.DataFrame(list(dictionary.token2id.items()), columns=["Token", "ID"])
            st.dataframe(dict_df)

            st.markdown("**🔖 Keyword in the topics (Gensim LDA):**")
            topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=True)
            st.text(topics)

            # --- pyLDAvis ---
            st.markdown("**📈 LDA Interactive Visualization:**")
            with st.spinner("Generating pyLDAvis visualization..."):
                vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
                html = pyLDAvis.prepared_data_to_html(vis_data)
                st.components.v1.html(html, width=1000, height=800)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a dataset to begin analysis.")
