# --- Libraries ---
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
#import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PIL import Image
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# --- Setup ---
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="centered", page_icon="💬")

# --- University Branding ---
col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("ucc.png")  # Ensure this file is in your project directory
    st.image(logo, width=80)
with col2:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)

st.markdown("---")

# --- Password Protection ---
PASSWORD = "CORBAN"  # Change this as needed
user_password = st.text_input("🔒 Enter Password to Access the App:", type="password")

if user_password != PASSWORD:
    st.warning("Please enter the correct password to continue.")
    st.stop()
    
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------ TEXT PROCESSING FUNCTIONS ------------------
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

# ------------------ STREAMLIT CONFIG ------------------
#st.set_page_config(page_title="Sentiment Typing App", layout="centered")

# Inject CSS for UCC branding and styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        h1 {
            color: #002147;
        }
        .university-header {
            background-color: #002147;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        .stButton>button {
            background-color: #002147;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stDownloadButton>button {
            background-color: #FFD700;
            color: black;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER & LOGO ------------------
st.image("ucc.png", use_container_width=False, width=150)
st.markdown('<div class="university-header"><h2>University of Cape Coast</h2><p>Sentiment Analysis Web App</p></div>', unsafe_allow_html=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Custom CSS ---
st.markdown("""
    <style>
        .main { background-color: #f4f6f9; }
        .university-header {
            background-color: #002147;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        .stButton>button { background-color: #002147; color: white; border-radius: 8px; font-size: 16px; }
        .stDownloadButton>button { background-color: #FFD700; color: black; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Centered Logo ---
st.image("ucc.png", use_container_width=False, width=150)
st.markdown('<div class="university-header"><h2>University of Cape Coast</h2><p>Sentiment & Topic Analysis Web App</p></div>', unsafe_allow_html=True)

# --- About ---
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Developed by **Bubutor Corban Enam** after participating in an NLP training session organized by **Professor Andy**.

    **Features:**
    - Sentiment Analysis (Polarity, Subjectivity, Opinion/Fact)
    - WordCloud Visualization
    - Sklearn & Gensim LDA Topic Modeling
    - pyLDAvis Interactive Topic Visualization
    - Supports CSV, Excel, and TXT files
    """)

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

# LDA Functions
def initialize_and_transform_dtms(df_comments):
    tf_vectorizer = CountVectorizer(
        strip_accents='unicode',
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        max_df=0.5,
        min_df=10
    )
    dtm_tf = tf_vectorizer.fit_transform(df_comments.values.astype('U'))

    tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
    dtm_tfidf = tfidf_vectorizer.fit_transform(df_comments.values.astype('U'))

    return tf_vectorizer, dtm_tf, tfidf_vectorizer, dtm_tfidf

def train_sklearn_lda_models(dtm_tf, dtm_tfidf, n_components=10, random_state=50):
    lda_tf = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda_tf.fit(dtm_tf)

    lda_tfidf = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda_tfidf.fit(dtm_tfidf)

    return lda_tf, lda_tfidf

def prepare_text_for_gensim(comments_list):
    extra_stopwords = ['from', 'subject', 're', 'edu', 'use']
    all_stopwords = stopwords.words('english') + extra_stopwords

    def sentences_to_words_generator(sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence), deacc=True))

    def remove_custom_stopwords(texts, custom_stopwords):
        return [[word for word in simple_preprocess(str(doc)) if word not in custom_stopwords] for doc in texts]

    comment_words = list(sentences_to_words_generator(comments_list))
    comment_words = remove_custom_stopwords(comment_words, all_stopwords)
    return comment_words

def create_gensim_corpus(comment_words):
    id2word = corpora.Dictionary(comment_words)
    corpus = [id2word.doc2bow(text) for text in comment_words]
    return id2word, corpus

def train_gensim_lda_model(corpus, id2word, num_topics=10):
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=50,
        passes=10,
        per_word_topics=True
    )
    return lda_model, lda_model[corpus]

# --- Session State ---
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Original Comment", "Cleaned Comment", "Polarity", "Subjectivity", "Sentiment", "Opinion/Fact"
    ])

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your file (CSV, Excel, or TXT)", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\t", engine='python')
        else:
            st.error("Unsupported file format.")
            df = None

        if df is not None:
            st.success("✅ File uploaded successfully!")
            text_cols = df.select_dtypes(include="object").columns.tolist()
            selected_col = st.selectbox("Select the comment column", text_cols)

            if st.button("🔎 Analyze Uploaded Comments"):
                batch_results = []
                for comment in df[selected_col].dropna():
                    cleaned = clean_text(comment)
                    polarity, subjectivity, sentiment, opinion_type = analyze_sentiment(cleaned)
                    batch_results.append({
                        "Original Comment": comment,
                        "Cleaned Comment": cleaned,
                        "Polarity": polarity,
                        "Subjectivity": subjectivity,
                        "Sentiment": sentiment,
                        "Opinion/Fact": opinion_type
                    })
                batch_df = pd.DataFrame(batch_results)
                st.session_state.results_df = batch_df

                # WordCloud
                all_text = " ".join(batch_df["Cleaned Comment"].dropna().tolist())
                if all_text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
                    st.markdown("### ☁️ Word Cloud for Uploaded Comments")
                    st.image(wordcloud.to_array(), use_container_width=True)

                st.markdown("### ✅ Batch Analysis Results")
                st.dataframe(batch_df)

                st.download_button(
                    "📥 Download Batch Results",
                    data=batch_df.to_csv(index=False).encode(),
                    file_name="batch_sentiment_results.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Visualization ---
if not st.session_state.results_df.empty:
    st.markdown("### 🗂️ Analysis History")
    st.dataframe(st.session_state.results_df)

    st.download_button(
        "📥 Download All Results",
        data=st.session_state.results_df.to_csv(index=False).encode(),
        file_name="all_sentiment_results.csv",
        mime="text/csv"
    )

    # Sentiment Distribution
    st.markdown("### 📊 Sentiment Distribution")
    sentiment_counts = st.session_state.results_df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    color_scale = alt.Scale(
        domain=["😊 Positive", "😐 Neutral", "😠 Negative"],
        range=["#2ECC71", "#B2BABB", "#E74C3C"]
    )

    bar_chart = alt.Chart(sentiment_counts).mark_bar().encode(
        x=alt.X("Sentiment", sort=["😊 Positive", "😐 Neutral", "😠 Negative"]),
        y="Count",
        color=alt.Color("Sentiment", scale=color_scale),
        tooltip=["Sentiment", "Count"]
    ).properties(width=600, height=400)

    st.altair_chart(bar_chart, use_container_width=True)

    # Scatter plot
    st.markdown("### 📌 Polarity vs Subjectivity")
    scatter = alt.Chart(st.session_state.results_df).mark_circle(size=70).encode(
        x='Polarity',
        y='Subjectivity',
        color='Sentiment',
        tooltip=['Original Comment', 'Polarity', 'Subjectivity', 'Sentiment']
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

    # --- Topic Modeling ---
    st.markdown("---")
    st.header("🧠 Advanced Analysis: Topic Modeling (LDA)")

    if st.checkbox("🔍 Perform Topic Modeling (LDA) on Cleaned Comments"):
        num_topics = st.slider("Select Number of Topics", min_value=3, max_value=15, value=5)

        cleaned_comments = st.session_state.results_df["Cleaned Comment"].dropna()

        # Sklearn LDA
        tf_vectorizer, dtm_tf, tfidf_vectorizer, dtm_tfidf = initialize_and_transform_dtms(cleaned_comments)
        lda_tf, lda_tfidf = train_sklearn_lda_models(dtm_tf, dtm_tfidf, n_components=num_topics)

        st.markdown("#### 🔖 Top Words per Topic (TF DTM):")
        tf_feature_names = tf_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda_tf.components_):
            top_features_ind = topic.argsort()[:-10 - 1:-1]
            top_features = [tf_feature_names[i] for i in top_features_ind]
            st.write(f"**Topic {topic_idx+1}:**", ", ".join(top_features))

        # Gensim LDA + pyLDAvis
        comment_words = prepare_text_for_gensim(cleaned_comments.tolist())
        id2word, corpus = create_gensim_corpus(comment_words)
        lda_model_gensim, doc_lda = train_gensim_lda_model(corpus, id2word, num_topics=num_topics)

        with st.spinner("Generating pyLDAvis visualization..."):
            vis_data = gensimvis.prepare(lda_model_gensim, corpus, id2word)
            html_string = pyLDAvis.prepared_data_to_html(vis_data)
            st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

# --- Single Comment Analysis ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("✍️ Analyze a New Comment")
    user_comment = st.text_area("Type your comment here 👇", height=150)

    if st.button("🔍 Analyze My Comment"):
        if user_comment.strip() == "":
            st.warning("Please enter a comment.")
        else:
            cleaned = clean_text(user_comment)
            polarity, subjectivity, sentiment, opinion_type = analyze_sentiment(cleaned)

            st.markdown("### ✨ Sentiment Result")
            st.markdown(f"📝 Your comment expresses a **{sentiment}** sentiment and is more **{opinion_type.lower()}-based**.")

            new_row = {
                "Original Comment": user_comment,
                "Cleaned Comment": cleaned,
                "Polarity": polarity,
                "Subjectivity": subjectivity,
                "Sentiment": sentiment,
                "Opinion/Fact": opinion_type
            }
            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

else:
    st.info("⚠️ Please upload a file to begin sentiment analysis.")
