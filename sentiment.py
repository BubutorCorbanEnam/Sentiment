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

# ------------------ PROFESSIONAL BACKGROUND ------------------
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy. This app allows users to analyze the sentiment of comments using natural language processing.
    
    It supports both batch analysis via CSV upload and manual typing. Results include polarity, subjectivity, sentiment type, and visual insights. Ideal for researchers, marketers, and educators.
    """)


# ----------------- Session State Password -----------------
#if "authenticated" not in st.session_state:
#    st.session_state["authenticated"] = False

#if not st.session_state["authenticated"]:
#    password = st.text_input("🔒 Enter Password:", type="password")
#    if password == "CORBAN":
#        st.session_state["authenticated"] = True
#        st.success("Access granted. Welcome!")
#    else:
#        st.stop()

# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# --- Functions for Text Analysis ---

def clean_text(text):
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing URLs, mentions, hashtags, and special characters
    - Tokenizing the text
    - Removing stopwords and single-character words
    - Lemmatizing the words
    """
    text = str(text).lower()
    # Remove URLs, mentions, hashtags, and non-alphabetic characters
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    # Filter out stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using TextBlob.
    Returns polarity, subjectivity, sentiment label, and opinion type.
    """
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    
    # Determine sentiment label based on polarity
    if polarity > 0:
        sentiment = "😊 Positive"
    elif polarity < 0:
        sentiment = "😠 Negative"
    else:
        sentiment = "😐 Neutral"
        
    # Determine if it's an opinion or fact based on subjectivity
    opinion = "Opinion" if subjectivity > 0 else "Fact"
    
    return polarity, subjectivity, sentiment, opinion

def generate_wordcloud(text):
    """
    Generates a WordCloud image from the given text.
    """
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words)
    return wc.generate(text)

def prepare_gensim_data(texts):
    """
    Prepares text data for Gensim LDA model by:
    - Adding custom stopwords
    - Simple preprocessing (de-accenting, tokenizing)
    - Filtering out custom stopwords
    """
    # Extend default stopwords with common LDA-specific ones
    custom_stopwords = stop_words.union({'from', 'subject', 're', 'edu', 'use'})
    processed_texts = [
        [word for word in simple_preprocess(str(doc), deacc=True) if word not in custom_stopwords]
        for doc in texts
    ]
    return processed_texts

@st.cache_resource(show_spinner=False)
def train_gensim_lda_model(_corpus, _id2word, num_topics):
    """
    Trains a Gensim LDA (Latent Dirichlet Allocation) model.
    Uses st.cache_resource to memoize the model for performance.
    """
    lda_model = gensim.models.LdaModel(
        corpus=_corpus,
        id2word=_id2word,
        num_topics=num_topics,
        random_state=50, # For reproducibility
        per_word_topics=True # To enable pyLDAvis visualization
    )
    return lda_model

# --- Streamlit Application Layout ---

#st.set_page_config(layout="wide", page_title="Text Analysis Tool")
#st.title("Text Analysis Tool 📝")
st.markdown("Upload your text data (CSV, Excel, or TXT) to perform sentiment analysis, generate word clouds, and discover topics using LDA.")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or TXT", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file:
    try:
        # Read the uploaded file into a DataFrame
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment"])
        else:
            st.error("🚨 Unsupported file format. Please upload a CSV, Excel, or TXT file.")
            st.stop()

        # Allow user to select the text column from the DataFrame
        text_cols = df.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.warning("⚠️ No text columns found in the uploaded file. Please ensure your file contains text data.")
            st.stop()
        
        selected_col = st.selectbox("Select Text Column for Analysis", text_cols)

        # --- Sentiment Analysis and Word Cloud Section ---
        if st.button("🔍 Run Sentiment Analysis & Word Cloud"):
            with st.spinner("Analyzing sentiment and generating word cloud..."):
                results = []
                # Process each comment for sentiment analysis
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

                # Download button for results
                st.download_button(
                    label="📥 Download Sentiment Results CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

            # --- Word Cloud Visualization ---
            st.markdown("---")
            st.subheader("☁️ Word Cloud")
            all_cleaned_text = " ".join(results_df["Cleaned Text"].tolist())
            if len(all_cleaned_text.strip()) > 0:
                wc_image = generate_wordcloud(all_cleaned_text)
                st.image(wc_image.to_array(), caption="Word Cloud of Cleaned Text", use_container_width=True)
            else:
                st.info("ℹ️ Not enough cleaned text available to generate a meaningful Word Cloud.")

            # --- Sentiment Distribution Chart ---
            st.markdown("---")
            st.subheader("📊 Sentiment Distribution")
            counts = results_df['Sentiment'].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort="-y", title="Sentiment Category"),
                y=alt.Y('Count', title="Number of Comments"),
                color=alt.Color('Sentiment', legend=None), # Color by sentiment, remove legend
                tooltip=['Sentiment', 'Count']
            ).properties(
                title="Distribution of Sentiment Categories"
            )
            st.altair_chart(chart, use_container_width=True)

            # --- Sentiment Scatter Plot ---
            st.markdown("---")
            st.subheader("🎯 Sentiment Scatter Plot")
            scatter_chart = alt.Chart(results_df).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X('Polarity', title='Polarity (Negative to Positive)'),
                y=alt.Y('Subjectivity', title='Subjectivity (Fact to Opinion)'),
                color=alt.Color('Sentiment', legend=alt.Legend(title="Sentiment")),
                tooltip=['Original Comment', 'Polarity', 'Subjectivity', 'Sentiment']
            ).interactive().properties(
                title="Polarity vs. Subjectivity of Comments"
            )
            st.altair_chart(scatter_chart, use_container_width=True)

        # --- LDA Topic Modeling Section ---
        st.markdown("---")
        st.header("🧠 Topic Modeling (Latent Dirichlet Allocation - LDA)")
        st.write("LDA helps identify underlying topics in your text data.")

        # Prepare data for LDA
        processed_texts_for_lda = prepare_gensim_data(df[selected_col].dropna().tolist())
        
        # Check if processed texts are suitable for LDA
        if not processed_texts_for_lda:
            st.warning("⚠️ No valid text found for topic modeling after cleaning. Please check your data.")
        else:
            id2word = corpora.Dictionary(processed_texts_for_lda)
            # Filter out tokens that appear in less than no_below documents
            # or more than no_above documents (fraction of total corpus size)
            id2word.filter_extremes(no_below=5, no_above=0.5) 
            corpus = [id2word.doc2bow(text) for text in processed_texts_for_lda]

            # Filter out empty documents from the corpus, which can occur after filtering extremes
            corpus = [doc for doc in corpus if doc] 

            # Ensure sufficient data for LDA to run meaningfully
            if len(corpus) < 3 or len(id2word) < 3: # Need at least 3 documents and 3 unique words
                st.warning("⚠️ Not enough unique documents or vocabulary for LDA topic modeling. Need at least 3 documents and 3 unique words after processing.")
            else:
                # Calculate max_topics dynamically based on data, no hardcoded limit
                # The maximum number of topics can't exceed the number of documents or the vocabulary size
                max_topics = min(len(corpus) -1, len(id2word)) # Subtract 1 from corpus length for safety if it's very small
                if max_topics < 3:
                     st.warning(f"⚠️ Insufficient data to create at least 3 topics after filtering. Only {max_topics} topics can be potentially generated.")
                else:
                    # Slider for user to select number of topics
                    num_topics = st.slider(
                        "Select Number of Topics for LDA (More topics can be harder to interpret)", 
                        min_value=3, 
                        max_value=20, 
                        value=min(5, max_topics), # Default to 5 or max_topics if smaller
                        step=1
                    )

                    if st.button("🚀 Run LDA Topic Analysis"):
                        with st.spinner(f"Training LDA model with {num_topics} topics... This might take a moment."):
                            try:
                                lda_model = train_gensim_lda_model(corpus, id2word, num_topics)

                                st.markdown("### 📊 Top Words Per Topic")
                                for idx, topic in lda_model.print_topics(num_words=10): # Show top 10 words per topic
                                    st.write(f"**Topic {idx+1}:** {topic}")

                                # pyLDAvis visualization
                                st.markdown("---")
                                st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                                st.info("ℹ️ This interactive chart might take a few seconds to load. It helps visualize topic relationships and word saliency.")
                                with st.spinner("Generating interactive visualization..."):
                                    if corpus and id2word: # Double check corpus and dictionary before creating vis
                                        vis = gensimvis.prepare(lda_model, corpus, id2word)
                                        html_string = pyLDAvis.prepared_data_to_html(vis)
                                        st.components.v1.html(html_string, width=1000, height=800, scrolling=True)
                                    else:
                                        st.warning("⚠️ Cannot generate pyLDAvis visualization: Corpus or dictionary is empty.")

                            except Exception as e:
                                st.error(f"An error occurred during LDA topic modeling: {e}")
                                st.warning("Please try reducing the number of topics or ensuring your dataset has enough diverse text.")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing your file: {e}")
        st.info("Please ensure your file is correctly formatted and the selected column contains text data.")

else:
    st.info("☝️ Please upload a dataset (CSV, Excel, or TXT) to begin your text analysis journey!")
