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
from scipy.spatial.distance import euclidean
import seaborn as sns
from pprint import pprint # Keep this here

# Import io and redirect_stdout here, once and at the top with other imports
import io
from contextlib import redirect_stdout
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
    This **AI & Data Science | Sentiment Analysis Web App** is a specialized tool designed for in-depth text analysis. Built by Bubutor Corban Enam following an enriching NLP training session organized by Professor Andy, its primary goal is to empower users with robust capabilities in **sentiment analysis** and **topic discovery**.

    ---

    ### Core Functionality

    The application offers a streamlined workflow for understanding textual data:

    * **File Upload**: Users can easily upload their text datasets in common formats like CSV, Excel, or plain TXT files.
    * **Text Preprocessing**: Before analysis, the app intelligently cleans the text data. This involves converting text to lowercase, removing URLs, mentions (`@username`), hashtags (`#hashtags`), and non-alphabetic characters. It then tokenizes the text, removes common **stopwords** (e.g., "the", "is"), and performs **lemmatization** (reducing words to their base form, e.g., "running" to "run") to standardize the vocabulary.
    * **Sentiment Analysis**: Leveraging the **TextBlob** library, the app quantifies the emotional tone of each piece of text. It provides:
        * **Polarity**: A score ranging from -1 (most negative) to +1 (most positive).
        * **Subjectivity**: A score from 0 (very objective/factual) to 1 (very subjective/opinionated).
        * **Categorical Sentiment**: Labels like "😊 Positive", "😠 Negative", or "😐 Neutral" for easy interpretation.
        * **Opinion Type**: Differentiates between "Opinion" and "Fact" based on subjectivity.
    * **Topic Modeling with LDA**: The app utilizes **Latent Dirichlet Allocation (LDA)**, a powerful unsupervised machine learning technique from the **Gensim** library, to identify hidden thematic structures (topics) within your text data. Users can select the desired number of topics, and the app will display the most prominent words associated with each.
    * **Custom Topic Labeling**: To enhance interpretability, users can assign their own meaningful names to the automatically identified topics, making the results more relevant to their specific domain or research.

    ---

    ### Data Visualization & Insights

    Beyond raw data, the app provides intuitive visualizations for a deeper understanding:

    * **Sentiment Analysis Results Table**: A comprehensive table displaying original text, cleaned text, and their respective sentiment metrics.
    * **Topic Analysis Plots**:
        * **Document Counts per Topic**: Bar charts illustrating the distribution of documents across the identified topics.
        * **Topic Polarity Distribution**: Stacked bar charts showing the percentage of positive, negative, and neutral sentiments within each specific topic, allowing for a granular view of sentiment by theme.
    * **Topic Relationship Graph**: This **NetworkX** graph visually represents the relationships between topics. Connections (edges) are drawn between topics, with their **thickness proportional to the similarity of their sentiment profiles**. This helps identify how different themes in your data are emotionally related. The similarity is calculated using **Euclidean distance** between the sentiment percentage profiles of each topic, transformed into a similarity score.
    * **Interactive LDA Visualization (pyLDAvis)**: An advanced interactive visualization that allows users to explore topics by showing their inter-topic distances and the most relevant terms within each, providing a highly navigable interface for topic interpretation.

    ---

    This application serves as a valuable resource for researchers, analysts, and anyone looking to extract meaningful sentiment and thematic insights from large volumes of text data.
    """)
# --- NLTK Setup ---
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# --- Text Processing Functions ---
def clean_text(text):
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing URLs (http/https and www)
    - Removing mentions (@username) and hashtags (#tag)
    - Removing non-alphabetic characters (keeping spaces)
    - Tokenizing the text
    - Lemmatizing tokens
    - Removing stopwords and single-character tokens
    Returns the cleaned and processed text as a string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)       # Remove mentions and hashtags
    text = re.sub(r"[^a-z\s]", "", text)       # Remove non-alphabetic characters
    
    tokens = word_tokenize(text)
    # Filter out stopwords and single-character tokens, then lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using TextBlob.
    Returns:
    - polarity (float): A float within the range [-1.0, 1.0] where -1 is negative and 1 is positive.
    - subjectivity (float): A float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    - sentiment (str): A categorical label ("😊 Positive", "😠 Negative", "😐 Neutral").
    - opinion (str): A simplified label ("Opinion" if subjective, "Fact" if objective).
    """
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
    """
    Trains a Gensim LDA model. This function is cached using st.cache_resource
    to prevent re-training on every Streamlit rerun, significantly improving performance.
    """
    with st.spinner(f"Training LDA model with {_num_topics} topics... This may take a moment."):
        lda_model = gensim.models.LdaModel(
            corpus=_corpus,
            id2word=_id2word,
            num_topics=_num_topics,
            random_state=50, # Set for reproducibility of results
            passes=10,        # Number of passes through the corpus during training
            iterations=50,    # Number of iterations for each document
            per_word_topics=True # Keep track of per-word topic probabilities
        )
    return lda_model

# --- Streamlit Session State Initialization ---
# Session state variables persist data across Streamlit reruns,
# preventing loss of analysis results when widgets are interacted with.
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
    st.session_state["num_topics"] = 5 # Default number of topics for the slider

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload CSV, Excel, or TXT File", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file:
    try:
        # Read the uploaded file based on its extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            # For TXT files, assume each line is a separate comment/document
            df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["comment"])
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, or TXT file.")
            st.stop() # Stop execution if the file format is not recognized

        st.success(f"File '{uploaded_file.name}' loaded successfully! Now select a text column.")

        # Identify and allow the user to select the text column for sentiment analysis
        text_cols = df.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.error("No text columns found in the uploaded file. Please ensure your file contains text data.")
            st.stop() # Stop if no text columns are found
        selected_col = st.selectbox("Select the Text Column for Sentiment Analysis", text_cols)

        # Button to trigger sentiment analysis
        if st.button("🔍 Run Sentiment Analysis"):
            if selected_col:
                with st.spinner("Performing sentiment analysis... This might take a moment."):
                    results = []
                    # Iterate through each comment in the selected column, dropping any NaN values
                    for comment_original in df[selected_col].dropna():
                        # Sentiment analysis is now performed on the ORIGINAL comment
                        polarity, subjectivity, sentiment, opinion = analyze_sentiment(comment_original)
                        # The text for LDA is still cleaned separately
                        cleaned_comment = clean_text(comment_original) 
                        results.append({
                            "Original": comment_original,
                            "Cleaned": cleaned_comment, # Store cleaned text for LDA
                            "Polarity": polarity,
                            "Subjectivity": subjectivity,
                            "Sentiment": sentiment,
                            "Type": opinion
                        })
                    sentiment_df = pd.DataFrame(results)
                    st.session_state["sentiment_df"] = sentiment_df # Store results in session state
                    st.success("Sentiment analysis complete!")

                    st.subheader("🗂️ Sentiment Analysis Results Table")
                    st.dataframe(sentiment_df) # Display the sentiment analysis results table
            else:
                st.warning("Please select a text column from the dropdown to run sentiment analysis.")

    except Exception as e:
        st.error(f"An error occurred during file processing or sentiment analysis: {e}. Please check your file format and content.")
        st.stop() # Stop execution on error

# --- Visualizations Section ---
if st.session_state["sentiment_df"] is not None and not st.session_state["sentiment_df"].empty:
    st.markdown("---")
    st.header("📊 Sentiment Visualizations")

    # --- Overall Word Cloud ---
    st.subheader("☁️ Overall Word Cloud")
    st.info("This word cloud shows the most frequent words across all cleaned text data, providing a quick visual summary of common terms.")
    
    all_words = ' '.join(st.session_state["sentiment_df"]['Cleaned'].dropna())
    
    if all_words.strip():  # Check if all_words is not just whitespace
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Overall Word Cloud')
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory
    else:
        st.warning("No meaningful cleaned text to generate the Overall Word Cloud.")
            
    st.markdown("---")

    # --- Sentiment Distribution Bar Plot ---
    st.subheader("📊 Overall Sentiment Distribution")
    st.info("This bar chart shows the total count of comments classified as Positive, Neutral, or Negative.")

    sentiment_counts = st.session_state["sentiment_df"]["Sentiment"].value_counts().reindex(["😊 Positive", "😐 Neutral", "😠 Negative"], fill_value=0)

    # Define colors for the bars
    bar_colors = ["green", "orange", "red"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', ax=ax, color=bar_colors)
    ax.set_title('Distribution of Sentiment Types')
    ax.set_xlabel('Sentiment Type')
    ax.set_ylabel('Number of Comments')
    ax.tick_params(axis='x', rotation=0) # No rotation needed for 3 bars
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory
            
    st.markdown("---")

    # --- Polarity vs. Subjectivity Scatter Plot ---
    st.subheader("📈 Polarity vs. Subjectivity Scatter Plot (Colored by Sentiment Type)")
    st.info("This plot helps visualize the distribution of comments based on their emotional tone (polarity) and factual basis (subjectivity), with points colored by their classified sentiment.")

    chart = alt.Chart(st.session_state["sentiment_df"]).mark_circle(size=60).encode(
        x=alt.X('Polarity', axis=alt.Axis(title='Polarity (-1: Negative, +1: Positive)')),
        y=alt.Y('Subjectivity', axis=alt.Axis(title='Subjectivity (0: Factual, 1: Opinion)')),
        color=alt.Color('Sentiment',
                        scale=alt.Scale(domain=["😊 Positive", "😠 Negative", "😐 Neutral"],
                                        range=['green', 'red', 'orange']),
                        legend=alt.Legend(title="Sentiment Type")),
        tooltip=['Original', 'Polarity', 'Subjectivity', 'Sentiment', 'Type']
    ).properties(
        title='Polarity vs. Subjectivity Scatter Plot'
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# --- LDA Topic Modeling Section ---
# This section only becomes active if sentiment analysis has been run and results are available
if st.session_state["sentiment_df"] is not None and not st.session_state["sentiment_df"].empty:
    st.markdown("---")
    st.header("🧠 LDA Topic Modeling (Manual Topic Assignment Supported)")

    # Prepare data for LDA: Filter out empty or whitespace-only cleaned comments
    initial_clean_comments = st.session_state["sentiment_df"]["Cleaned"].dropna()
    clean_comments_for_lda = [comment for comment in initial_clean_comments if comment.strip()]

    if not clean_comments_for_lda:
        st.warning("No valid cleaned text data available for Topic Modeling after initial filtering. Please ensure your data contains meaningful text.")
    else:
        processed_texts = [simple_preprocess(doc) for doc in clean_comments_for_lda]
        final_processed_texts = [text for text in processed_texts if text]

        if not final_processed_texts:
            st.warning("No valid processed text data for Topic Modeling after detailed preprocessing. This might happen if all comments consist only of stopwords or short words.")
        else:
            id2word = corpora.Dictionary(final_processed_texts)
            id2word.filter_extremes(no_below=5, no_above=0.5)
            corpus = [id2word.doc2bow(text) for text in final_processed_texts]

            st.session_state["id2word"] = id2word
            st.session_state["corpus"] = corpus

            # Get the number of topics from the slider
            num_topics_from_slider = st.slider("Select Number of Topics", 3, 20, st.session_state["num_topics"], key="num_topics_slider")
            
            # --- MODIFICATION START (Ensuring consistency with num_topics) ---
            # Check if the slider value has changed from the last recorded session state value
            if num_topics_from_slider != st.session_state["num_topics"]:
                # If it has changed, update the session state and clear related cached items
                st.session_state["num_topics"] = num_topics_from_slider
                st.session_state["topic_labels"] = {} # Clear old labels
                st.session_state["lda_model"] = None # Invalidate existing model to force retraining

                # Clear the cache for the train_gensim_lda_model function
                # This is the most direct way to ensure the cached model is re-generated.
                train_gensim_lda_model.clear() 
                st.rerun() # Rerun the app to reflect the cleared state and prompt user
            
            # Add an info message to guide the user to retrain the model
            # This message will appear if the model is None (due to initial state or invalidation)
            # or if the current model's topic count doesn't match the slider's.
            if st.session_state["lda_model"] is None:
                st.info("Please click '🚀 Run LDA Model Training' to apply the selected number of topics.")
            elif st.session_state["lda_model"].num_topics != st.session_state["num_topics"]:
                st.info(f"The current model was trained with {st.session_state['lda_model'].num_topics} topics. Click '🚀 Run LDA Model Training' to retrain with {st.session_state['num_topics']} topics.")
            # --- MODIFICATION END ---


            if st.button("🚀 Run LDA Model Training"):
                if not corpus:
                    st.error("Cannot run LDA: The corpus is empty. This usually means no meaningful text was found after preprocessing or filtering.")
                else:
                    # Train the model with the user's selected num_topics ONLY
                    # This call will now definitely re-run if the cache was cleared.
                    lda_model = train_gensim_lda_model(corpus, id2word, st.session_state["num_topics"])
                    st.session_state["lda_model"] = lda_model
                    st.success("LDA model training complete!")

                    st.subheader("🔑 Top Words per Topic (from LDA Model)")
                    # Display top words for each topic, ensuring it's based on num_topics
                    # Get the actual topic IDs from the newly trained model (which respects num_topics)
                    # This list will contain exactly 'num_topics' elements.
                    actual_topic_ids_from_model_training = [topic[0] for topic in lda_model.show_topics(num_topics=st.session_state["num_topics"], formatted=False)]
                    
                    # Initialize/update topic_labels with default labels for the *actual* topic IDs from the model
                    # This ensures that even if Gensim gives non-sequential IDs, we prepare labels for them.
                    st.session_state["topic_labels"] = {} # Ensure a fresh start for labels
                    for idx in actual_topic_ids_from_model_training:
                        st.session_state["topic_labels"][idx] = f"Topic {idx+1}"
                    
                    # Now display using the actual IDs and their (potentially customized) labels
                    for i, idx in enumerate(actual_topic_ids_from_model_training): # Use enumerate to get a sequential display number
                        words = ", ".join([w for w, p in lda_model.show_topic(idx, topn=30)]) # Use show_topic for a specific index
                        current_display_label = st.session_state["topic_labels"].get(idx, f"Topic {idx+1}")
                        st.write(f"**{current_display_label} (Internal ID: {idx+1}):** {words}") # Show internal ID for clarity
                    
                    st.subheader("Raw LDA Topics for Expert Review")
                    f = io.StringIO()
                    with redirect_stdout(f):
                        # Ensure pprint also uses the correct num_topics
                        pprint(lda_model.print_topics(num_topics=st.session_state["num_topics"], num_words=30))
                    s = f.getvalue()
                    st.code(s)

            if st.session_state["lda_model"]:
                st.markdown("---")
                st.subheader("📝 Assign Custom Labels to Topics")

                with st.form("topic_label_form"):
                    # Get the actual topic IDs from the currently trained model for consistent labeling
                    # This ensures the input fields match the topics generated by the model.
                    # This list will contain exactly 'num_topics' elements.
                    current_model_topic_ids = [topic[0] for topic in st.session_state["lda_model"].show_topics(num_topics=st.session_state["num_topics"], formatted=False)]
                    
                    # Update topic_labels to ensure only labels for current topics are kept
                    # and remove any labels for topics that are no longer present (from a previous run with more topics)
                    st.session_state["topic_labels"] = {
                        k: v for k, v in st.session_state["topic_labels"].items() if k in current_model_topic_ids
                    }
                    
                    # Ensure all current topics have a default label if not already set
                    for i in current_model_topic_ids:
                        if i not in st.session_state["topic_labels"]:
                            st.session_state["topic_labels"][i] = f"Topic {i+1}"

                    # Display text input fields for each topic based on the actual topic IDs
                    for i, topic_id_from_model in enumerate(current_model_topic_ids):
                        current_label = st.session_state["topic_labels"].get(topic_id_from_model, f"Topic {topic_id_from_model+1}")
                        new_label = st.text_input(f"Label for Topic {i+1} (Internal ID: {topic_id_from_model+1})",
                                                 value=current_label, key=f"topic_input_{topic_id_from_model}")
                        st.session_state["topic_labels"][topic_id_from_model] = new_label
                    
                    submit_labels = st.form_submit_button("✔️ Apply Topic Labels and Analyze")

                if submit_labels:
                    df_with_topics = st.session_state["sentiment_df"].copy()
                    full_topic_assignments = ["Unassigned"] * len(df_with_topics)

                    for idx, row in df_with_topics.iterrows():
                        cleaned_text_doc = row["Cleaned"]
                        
                        if pd.isna(cleaned_text_doc) or not cleaned_text_doc.strip():
                            full_topic_assignments[idx] = "Unassigned"
                            continue

                        doc_tokens = simple_preprocess(cleaned_text_doc)
                        
                        if not doc_tokens:
                            full_topic_assignments[idx] = "Unassigned"
                            continue

                        bow_for_this_doc = st.session_state["id2word"].doc2bow(doc_tokens)
                        
                        if not bow_for_this_doc:
                            full_topic_assignments[idx] = "Unassigned"
                            continue

                        try:
                            # Get document topics from the *currently trained* LDA model
                            topic_probs = st.session_state["lda_model"].get_document_topics(bow_for_this_doc)
                        except IndexError:
                            st.warning(f"Could not assign topic to document at index {idx} due to vocabulary mismatch. Assigning 'Unassigned'.")
                            full_topic_assignments[idx] = "Unassigned"
                            continue

                        if topic_probs:
                            assigned_topic_id = max(topic_probs, key=lambda x: x[1])[0]
                            # Use the actual topic_id from Gensim for lookup in topic_labels
                            label = st.session_state["topic_labels"].get(assigned_topic_id, f"Topic {assigned_topic_id+1}")
                            full_topic_assignments[idx] = label
                        else:
                            full_topic_assignments[idx] = "Unassigned"

                    df_with_topics["Topic"] = full_topic_assignments

                    # Filter out 'Unassigned' topics and ensure only topics from the *current* model are considered
                    # This ensures the plots only show the number of topics the model was trained with.
                    current_valid_labels = set(st.session_state["topic_labels"].values())
                    df_for_viz = df_with_topics[
                        (df_with_topics["Topic"] != "Unassigned") & 
                        (df_with_topics["Topic"].isin(current_valid_labels))
                    ].copy()

                    if not df_for_viz.empty:
                        st.subheader("📊 Topic Analysis: Document Counts per Topic")
                        plt.figure(figsize=(12, 7))
                        # Use value_counts().index to ensure bars are ordered by frequency
                        # This order will now only include the 'num_topics' topics.
                        sns.countplot(data=df_for_viz, x="Topic", palette="viridis", order=df_for_viz['Topic'].value_counts().index)
                        plt.title('Distribution of Documents Across Topics')
                        plt.xlabel('Topic Labels')
                        plt.ylabel('Number of Documents')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.close()

                        st.subheader("📊 Topic Polarity Distribution")
                        df_topic_polarity = df_for_viz.groupby('Topic')['Sentiment'].value_counts(normalize=True).unstack(fill_value=0)
                        
                        sentiment_order = ["😠 Negative", "😐 Neutral", "😊 Positive"]
                        for s_type in sentiment_order:
                            if s_type not in df_topic_polarity.columns:
                                df_topic_polarity[s_type] = 0.0
                        df_topic_polarity = df_topic_polarity[sentiment_order] * 100

                        color_mapping = {"😠 Negative": 'red', "😐 Neutral": 'orange', "😊 Positive": 'green'}
                        colors = [color_mapping[col] for col in df_topic_polarity.columns]

                        ax = df_topic_polarity.plot(kind='bar', color=colors, stacked=True, figsize=(12, 7))
                        ax.set_xlabel('Topic')
                        ax.set_ylabel('Percentage of Polarity (%)')
                        ax.set_title('Topic Polarity Distribution')
                        ax.set_xticklabels(df_topic_polarity.index, rotation=45, ha='right')
                        ax.legend(title='Polarity', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        st.pyplot(ax.get_figure())
                        plt.close()

                        st.subheader("🔗 Topic Relationship Graph (Based on Sentiment Profiles)")
                        st.info("This graph visualizes the relationships between topics. The thickness of a connection indicates the strength of the similarity in their sentiment profiles.")

                        df_sim_ready = df_topic_polarity.fillna(0)
                        
                        if df_sim_ready.empty:
                            st.warning("Cannot generate topic relationship graph: No data for sentiment similarity calculation after filtering.")
                        elif len(df_sim_ready) < 2:
                            st.info("Not enough topics (at least 2 required) to draw a relationship graph.")
                        else:
                            topic_names = df_sim_ready.index.tolist()
                            
                            topic_polarity_matrix = np.zeros((len(topic_names), len(topic_names)))
                            
                            for i in range(len(topic_names)):
                                for j in range(i + 1, len(topic_names)):
                                    profile_i = df_sim_ready.iloc[i].values.flatten()
                                    profile_j = df_sim_ready.iloc[j].values.flatten()

                                    dist = euclidean(profile_i, profile_j)
                                    similarity = 1 / (1 + dist)
                                    topic_polarity_matrix[i][j] = similarity
                                    topic_polarity_matrix[j][i] = similarity

                            G = nx.Graph()
                            G.add_nodes_from(topic_names)

                            for i in range(len(topic_polarity_matrix)):
                                for j in range(i + 1, len(topic_polarity_matrix[0])):
                                    G.add_edge(topic_names[i], topic_names[j], weight=topic_polarity_matrix[i][j])
                            
                            pos = nx.spring_layout(G)
                            plt.figure(figsize=(12, 8))

                            edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
                            
                            min_line_width = 0.5
                            max_line_width = 5.0

                            if edge_weights:
                                if max(edge_weights) == min(edge_weights):
                                    widths = [min_line_width] * len(edge_weights)
                                else:
                                    widths = [min_line_width + (w - min(edge_weights)) / (max(edge_weights) - min(edge_weights)) * (max_line_width - min_line_width) for w in edge_weights]
                            else:
                                widths = []

                            nx.draw(G, pos, with_labels=True, font_weight='bold',
                                    node_color='skyblue', node_size=2000, edge_color='gray',
                                    width=widths, alpha=0.8, font_size=10) 

                            edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
                            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.3) 

                            plt.title('Topic Similarity Graph (Based on Sentiment Profiles)')
                            plt.axis('off')
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.close()
                                    
                        st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                        st.info("This interactive visualization helps explore topics by showing their relationships and the most relevant terms. Move the mouse over topics and words for details.")
                        
                        if st.session_state["lda_model"] and st.session_state["corpus"] and st.session_state["id2word"]:
                            try:
                                # pyLDAvis preparation will use the already trained lda_model,
                                # which was trained with the user-selected num_topics.
                                vis = gensimvis.prepare(st.session_state["lda_model"], st.session_state["corpus"], st.session_state["id2word"], mds='mmds')
                                html_string = pyLDAvis.prepared_data_to_html(vis)
                                st.components.v1.html(html_string, width=1000, height=800, scrolling=True)
                            except Exception as e:
                                st.error(f"Error generating interactive LDA visualization: {e}. This can happen if the model or data is not suitable for visualization (e.g., too few topics or documents, or issues with the underlying data for MDS).")
                        else:
                            st.warning("LDA model or its components are not ready for interactive visualization. Please run LDA training first.")
                    else:
                        st.warning("No data available for topic analysis after filtering 'Unassigned' comments. Please ensure your original data has comments that can be assigned to topics.")
            else:
                st.info("Please run the LDA Model Training first to enable topic assignment and analysis.")
else:
    st.info("Upload your data and run sentiment analysis first to enable topic modeling and further analysis.")
