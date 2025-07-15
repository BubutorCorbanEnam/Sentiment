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
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PIL import Image
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import os

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="centered", page_icon="💬")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main { background-color: #f4f6f9; }
        h1, h2, h3, h4 { color: #002147; }
        .university-header {
            background-color: #002147;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }
        .stButton>button { background-color: #002147; color: white; }
        .stButton>button:hover { background-color: #0E4D92; }
        .stDownloadButton>button { background-color: #FFD700; color: black; }
        .stDownloadButton>button:hover { background-color: #E6C200; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

# --- NLTK Downloads ---
@st.cache_resource
def download_nltk_data():
    for item in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
        except LookupError:
            nltk.download(item)
    return True

download_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- University Branding ---
if os.path.exists("ucc.png"):
    col1, col2 = st.columns([1, 8])
    with col1:
        logo = Image.open("ucc.png")
        st.image(logo, width=80)
    with col2:
        st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='color:#0E4D92; font-weight:bold;'>University of Cape Coast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#555;'>AI & Data Science | Sentiment Analysis Web App</h4>", unsafe_allow_html=True)
    st.warning("`ucc.png` not found. Please ensure the image file is in the same directory as the script.")

st.markdown("---")

# --- Password Protection ---
PASSWORD = "CORBAN"
user_password = st.text_input("🔒 Enter Password to Access the App:", type="password")
if user_password != PASSWORD:
    st.warning("Please enter the correct password to continue.")
    st.stop()

st.markdown('<div class="university-header"><h2>University of Cape Coast</h2><p>Sentiment & Topic Analysis Web App</p></div>', unsafe_allow_html=True)

# --- About Section ---
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

# --- Text Processing Functions ---
@st.cache_data
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-zA-Z\s]", "", str(text).lower())
    tokens = [lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

@st.cache_data
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    sentiment = "😊 Positive" if polarity > 0 else "😠 Negative" if polarity < 0 else "😐 Neutral"
    opinion_type = "Opinion" if subjectivity > 0 else "Fact"
    return polarity, subjectivity, sentiment, opinion_type

# --- LDA Utility Functions ---
@st.cache_data
def initialize_and_transform_dtms(df_comments):
    tf_vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english', lowercase=True,
                                     token_pattern=r'\b[a-zA-Z]{3,}\b', max_df=0.5, min_df=10)
    dtm_tf = tf_vectorizer.fit_transform(df_comments.values.astype('U'))
    tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
    dtm_tfidf = tfidf_vectorizer.fit_transform(df_comments.values.astype('U'))
    return tf_vectorizer, dtm_tf, tfidf_vectorizer, dtm_tfidf

@st.cache_data
def train_sklearn_lda_models(dtm_tf, dtm_tfidf, n_components=10, random_state=50):
    lda_tf = LatentDirichletAllocation(n_components=n_components, random_state=random_state).fit(dtm_tf)
    lda_tfidf = LatentDirichletAllocation(n_components=n_components, random_state=random_state).fit(dtm_tfidf)
    return lda_tf, lda_tfidf

@st.cache_data
def prepare_text_for_gensim(comments_list):
    all_stopwords = stopwords.words('english') + ['from', 'subject', 're', 'edu', 'use']
    texts = [[word for word in simple_preprocess(str(doc)) if word not in all_stopwords] for doc in comments_list]
    return texts

@st.cache_data
def create_gensim_corpus(comment_words):
    id2word = corpora.Dictionary(comment_words)
    corpus = [id2word.doc2bow(text) for text in comment_words]
    return id2word, corpus

@st.cache_data
def train_gensim_lda_model(corpus, id2word, num_topics=10):
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                           random_state=50, passes=10, per_word_topics=True)
    return lda_model, lda_model[corpus]

# --- Session State Init ---
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Original Comment", "Cleaned Comment", "Polarity", "Subjectivity", "Sentiment", "Opinion/Fact"])

# --- Upload File ---
uploaded_file = st.file_uploader("📂 Upload your file (CSV, Excel, or TXT)", type=["csv", "xlsx", "xls", "txt"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") \
            else pd.read_excel(uploaded_file) if uploaded_file.name.endswith((".xlsx", ".xls")) \
            else pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment"])

        text_cols = df.select_dtypes(include="object").columns.tolist()
        selected_col = st.selectbox("Select the comment column", text_cols)

        if st.button("🔎 Analyze Uploaded Comments"):
            results = []
            for comment in df[selected_col].dropna():
                cleaned = clean_text(comment)
                polarity, subjectivity, sentiment, opinion_type = analyze_sentiment(cleaned)
                results.append({"Original Comment": comment, "Cleaned Comment": cleaned,
                                "Polarity": polarity, "Subjectivity": subjectivity,
                                "Sentiment": sentiment, "Opinion/Fact": opinion_type})
            result_df = pd.DataFrame(results)
            st.session_state.results_df = result_df

            st.dataframe(result_df)
            st.download_button("📥 Download Results", data=result_df.to_csv(index=False).encode(), file_name="sentiment_results.csv", mime="text/csv")

            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(result_df["Cleaned Comment"]))
            st.image(wordcloud.to_array(), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# --- Visualizations & LDA ---
if not st.session_state.results_df.empty:
    st.markdown("### 📊 Sentiment Distribution")
    sentiment_counts = st.session_state.results_df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    color_scale = alt.Scale(domain=["😊 Positive", "😐 Neutral", "😠 Negative"], range=["#2ECC71", "#B2BABB", "#E74C3C"])
    st.altair_chart(
        alt.Chart(sentiment_counts).mark_bar().encode(
            x=alt.X("Sentiment", sort=["😊 Positive", "😐 Neutral", "😠 Negative"]),
            y="Count", color=alt.Color("Sentiment", scale=color_scale),
            tooltip=["Sentiment", "Count"]).properties(width=600, height=400), use_container_width=True)

    st.markdown("### 📌 Polarity vs Subjectivity")
    st.altair_chart(
        alt.Chart(st.session_state.results_df).mark_circle(size=70).encode(
            x='Polarity', y='Subjectivity', color='Sentiment',
            tooltip=['Original Comment', 'Polarity', 'Subjectivity', 'Sentiment']).interactive(),
        use_container_width=True)

    if st.checkbox("🔍 Perform Topic Modeling (LDA)"):
        num_topics = st.slider("Number of Topics", 3, 15, 5)
        comments = st.session_state.results_df["Cleaned Comment"].dropna()

        tf_vectorizer, dtm_tf, tfidf_vectorizer, dtm_tfidf = initialize_and_transform_dtms(comments)
        lda_tf, lda_tfidf = train_sklearn_lda_models(dtm_tf, dtm_tfidf, n_components=num_topics)

        st.markdown("#### 🔖 Top Words per Topic (TF DTM):")
        tf_feature_names = tf_vectorizer.get_feature_names_out()
        for i, topic in enumerate(lda_tf.components_):
            words = ", ".join([tf_feature_names[j] for j in topic.argsort()[:-11:-1]])
            st.write(f"**Topic {i + 1}:**", words)

        comment_words = prepare_text_for_gensim(comments.tolist())
        id2word, corpus = create_gensim_corpus(comment_words)
        lda_model, doc_lda = train_gensim_lda_model(corpus, id2word, num_topics=num_topics)

        with st.spinner("Generating interactive topic visualization..."):
            vis_data = gensimvis.prepare(lda_model, corpus, id2word)
            html_string = pyLDAvis.prepared_data_to_html(vis_data)
            st.components.v1.html(html_string, width=1000, height=800, scrolling=True)

# --- Single Comment Analysis ---
st.markdown("---")
st.subheader("✍️ Analyze a New Comment")
user_comment = st.text_area("Type your comment here 👇", height=150)
if st.button("🔍 Analyze My Comment"):
    if not user_comment.strip():
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(user_comment)
        polarity, subjectivity, sentiment, opinion_type = analyze_sentiment(cleaned)
        st.markdown(f"📝 Your comment expresses a **{sentiment}** sentiment and is more **{opinion_type.lower()}-based**.")
        st.info(f"Polarity: {polarity}, Subjectivity: {subjectivity}")

        new_row = pd.DataFrame([{
            "Original Comment": user_comment,
            "Cleaned Comment": cleaned,
            "Polarity": polarity,
            "Subjectivity": subjectivity,
            "Sentiment": sentiment,
            "Opinion/Fact": opinion_type
        }])
        st.session_state.results_df = pd.concat([st.session_state.results_df, new_row], ignore_index=True)
        st.success("Comment added to analysis history!")
