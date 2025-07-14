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
import os # Import os for checking file existence

# --- Streamlit Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="UCC Sentiment Analysis Portal", layout="centered", page_icon="💬")

# --- Custom CSS (Consolidated and placed early) ---
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        h1, h2, h3, h4 {
            color: #002147; /* Dark blue for headers */
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
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0E4D92; /* Lighter blue on hover */
        }
        .stDownloadButton>button {
            background-color: #FFD700; /* Gold */
            color: black;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #E6C200; /* Darker gold on hover */
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .stAlert {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- NLTK Downloads (Cached for efficiency - runs only once per deployment) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt') # Corrected from 'punkt_tab'
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet', quiet=True)
    return True

if not download_nltk_data():
    st.error("Failed to download NLTK data. Please check your internet connection.")
    st.stop()

# Initialize NLTK resources once globally after download
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- University Branding (Consolidated and placed after initial setup) ---
# Ensure 'ucc.png' is in the same directory as your app.py for deployment
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
PASSWORD = "CORBAN"  # Change this as needed
user_password = st.text_input("🔒 Enter Password to Access the App:", type="password")

if user_password != PASSWORD:
    st.warning("Please enter the correct password to continue.")
    st.stop() # Halts execution if password is incorrect

# --- Main Header/Title (Consolidated) ---
# This serves as the main title of the app after password entry
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

# ------------------ TEXT PROCESSING FUNCTIONS (Defined once and cached) ------------------
@st.cache_data # Cache the cleaning function for performance
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

@st.cache_data # Cache the sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    sentiment = "😊 Positive" if polarity > 0 else "😠 Negative" if polarity < 0 else "😐 Neutral"
    opinion_type = "Opinion" if subjectivity > 0 else "Fact"
    return polarity, subjectivity, sentiment, opinion_type

# --- LDA Functions (Defined once and cached for performance on repeated runs) ---
@st.cache_data
def initialize_and_transform_dtms(df_comments):
    if df_comments.empty:
        return None, None, None, None

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

@st.cache_data
def train_sklearn_lda_models(dtm_tf, dtm_tfidf, n_components=10, random_state=50):
    if dtm_tf is None or dtm_tfidf is None:
        return None, None

    lda_tf = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda_tf.fit(dtm_tf)

    lda_tfidf = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda_tfidf.fit(dtm_tfidf)

    return lda_tf, lda_tfidf

@st.cache_data
def prepare_text_for_gensim(comments_list):
    if not comments_list:
        return []

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

@st.cache_data
def create_gensim_corpus(comment_words):
    if not comment_words:
        return None, None

    id2word = corpora.Dictionary(comment_words)
    corpus = [id2word.doc2bow(text) for text in comment_words]
    return id2word, corpus

@st.cache_data
def train_gensim_lda_model(corpus, id2word, num_topics=10):
    if corpus is None or id2word is None or not corpus:
        return None, None

    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=50,
        passes=10,
        per_word_topics=True
    )
    return lda_model, lda_model[corpus]

# --- Session State Initialization ---
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Original Comment", "Cleaned Comment", "Polarity", "Subjectivity", "Sentiment", "Opinion/Fact"
    ])

# --- File Upload Section ---
st.subheader("📂 Upload and Analyze Comments")
uploaded_file = st.file_uploader("Upload your file (CSV, Excel, or TXT)", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            # Assuming TXT files are line-separated comments, or single column tab-separated
            # Adjust delimiter if your TXT format is different
            df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment_text"])
        else:
            st.error("Unsupported file format.")
            df = None

        if df is not None:
            st.success("✅ File uploaded successfully!")
            # Automatically detect a suitable text column
            potential_text_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 1]
            if potential_text_cols:
                selected_col = st.selectbox("Select the comment column", potential_text_cols)
            else:
                st.error("No suitable text columns found in the uploaded file.")
                df = None # Invalidate df if no text column

            if df is not None and st.button("🔎 Analyze Uploaded Comments"):
                with st.spinner("Analyzing comments... This may take a while for large files."):
                    batch_results = []
                    # Ensure the selected column exists and is not empty
                    if selected_col in df.columns and not df[selected_col].dropna().empty:
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
                        st.session_state.results_df = batch_df # Update session state with new batch

                        # WordCloud
                        all_text = " ".join(batch_df["Cleaned Comment"].dropna().tolist())
                        if all_text.strip():
                            st.markdown("### ☁️ Word Cloud for Uploaded Comments")
                            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
                            st.image(wordcloud.to_array(), use_container_width=True)
                        else:
                            st.info("No cleaned text available to generate Word Cloud.")

                        st.markdown("### ✅ Batch Analysis Results")
                        st.dataframe(batch_df)

                        st.download_button(
                            "📥 Download Batch Results",
                            data=batch_df.to_csv(index=False).encode(),
                            file_name="batch_sentiment_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("The selected column is empty or does not exist. No comments to analyze.")
        else:
            st.info("Please upload a valid file to proceed with analysis.")

    except Exception as e:
        st.error(f"Error processing file: {e}. Please ensure the file format is correct and contains valid data.")
        st.exception(e) # Display full traceback for debugging


# --- Display Analysis History & Visualizations ---
if not st.session_state.results_df.empty:
    st.markdown("---")
    st.subheader("🗂️ Analysis History & Visualizations")
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

    # --- Topic Modeling Section ---
    st.markdown("---")
    st.header("🧠 Advanced Analysis: Topic Modeling (LDA)")

    if st.checkbox("🔍 Perform Topic Modeling (LDA) on Cleaned Comments"):
        cleaned_comments_for_lda = st.session_state.results_df["Cleaned Comment"].dropna()

        if cleaned_comments_for_lda.empty:
            st.warning("No cleaned comments available for Topic Modeling. Please upload a file and analyze comments first.")
        else:
            num_topics = st.slider("Select Number of Topics", min_value=3, max_value=15, value=5, key="num_topics_slider")

            with st.spinner("Performing LDA Topic Modeling..."):
                # Sklearn LDA
                tf_vectorizer, dtm_tf, tfidf_vectorizer, dtm_tfidf = initialize_and_transform_dtms(cleaned_comments_for_lda)

                if dtm_tf is not None and dtm_tfidf is not None:
                    lda_tf, lda_tfidf = train_sklearn_lda_models(dtm_tf, dtm_tfidf, n_components=num_topics)

                    st.markdown("#### 🔖 Top Words per Topic (TF DTM):")
                    tf_feature_names = tf_vectorizer.get_feature_names_out()
                    for topic_idx, topic in enumerate(lda_tf.components_):
                        top_features_ind = topic.argsort()[:-10 - 1:-1]
                        top_features = [tf_feature_names[i] for i in top_features_ind]
                        st.write(f"**Topic {topic_idx+1}:**", ", ".join(top_features))

                    # Gensim LDA + pyLDAvis
                    comment_words = prepare_text_for_gensim(cleaned_comments_for_lda.tolist())
                    id2word, corpus = create_gensim_corpus(comment_words)

                    if corpus is not None and id2word is not None and corpus: # Check if corpus is not empty
                        lda_model_gensim, doc_lda = train_gensim_lda_model(corpus, id2word, num_topics=num_topics)

                        if lda_model_gensim:
                            with st.spinner("Generating pyLDAvis visualization..."):
                                vis_data = gensimvis.prepare(lda_model_gensim, corpus, id2word)
                                html_string = pyLDAvis.prepared_data_to_html(vis_data)
                                st.components.v1.html(html_string, width=1000, height=800, scrolling=True)
                        else:
                            st.warning("Gensim LDA model could not be trained. Check data quality or number of topics.")
                    else:
                        st.warning("Gensim corpus could not be created. This might be due to very sparse or empty cleaned comments after stopword removal.")
                else:
                    st.warning("DTMs could not be created for topic modeling. This might be due to insufficient data or words after cleaning.")


# --- Single Comment Analysis Section ---
st.markdown("---")
st.subheader("✍️ Analyze a New Comment")
user_comment = st.text_area("Type your comment here 👇", height=150, key="single_comment_input")

if st.button("🔍 Analyze My Comment", key="analyze_single_comment_button"):
    if user_comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(user_comment)
        if not cleaned.strip(): # Check if cleaned text is empty
            st.warning("The comment could not be processed into meaningful words after cleaning. Please try a different comment.")
        else:
            polarity, subjectivity, sentiment, opinion_type = analyze_sentiment(cleaned)

            st.markdown("### ✨ Sentiment Result")
            st.markdown(f"📝 Your comment expresses a **{sentiment}** sentiment and is more **{opinion_type.lower()}-based**.")
            st.info(f"Polarity: {polarity} (closer to 1 is positive, -1 is negative, 0 is neutral)")
            st.info(f"Subjectivity: {subjectivity} (closer to 1 is opinion, 0 is fact)")

            new_row = {
                "Original Comment": user_comment,
                "Cleaned Comment": cleaned,
                "Polarity": polarity,
                "Subjectivity": subjectivity,
                "Sentiment": sentiment,
                "Opinion/Fact": opinion_type
            }
            # Append new row using pd.concat for immutability and proper session state update
            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.success("Comment added to analysis history!")

# Initial message if no data is present at all
if st.session_state.results_df.empty and uploaded_file is None:
    st.info("Upload a file or type a comment to begin sentiment analysis.")

