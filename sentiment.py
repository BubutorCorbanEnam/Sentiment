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
    # This prevents issues with gensim.corpora.Dictionary and doc2bow for empty strings.
    initial_clean_comments = st.session_state["sentiment_df"]["Cleaned"].dropna()
    clean_comments_for_lda = [comment for comment in initial_clean_comments if comment.strip()]

    if not clean_comments_for_lda:
        st.warning("No valid cleaned text data available for Topic Modeling after initial filtering. Please ensure your data contains meaningful text.")
        # st.stop() # Removed st.stop() to allow other parts of the app to render
    else: # Only proceed with LDA if there's valid data
        # Process texts into tokens for Gensim LDA
        processed_texts = [simple_preprocess(doc) for doc in clean_comments_for_lda]
        
        # Filter out any lists that became empty after simple_preprocess (e.g., if only stopwords or very short words were present)
        final_processed_texts = [text for text in processed_texts if text]

        if not final_processed_texts:
            st.warning("No valid processed text data for Topic Modeling after detailed preprocessing. This might happen if all comments consist only of stopwords or short words.")
            # st.stop() # Removed st.stop()
        else: # Only proceed with LDA if there's valid processed data
            # Create Gensim dictionary and corpus from the final processed texts
            id2word = corpora.Dictionary(final_processed_texts)
            
            # Filter out words that appear in too few (no_below) or too many (no_above) documents
            # These parameters can be tuned to improve topic quality.
            id2word.filter_extremes(no_below=5, no_above=0.5) # Example: min 5 documents, max 50% of documents
            
            corpus = [id2word.doc2bow(text) for text in final_processed_texts]

            # Update session state for LDA components (used for pyLDAvis and initial topic display)
            st.session_state["id2word"] = id2word
            st.session_state["corpus"] = corpus

            # Slider to select the number of topics for LDA
            num_topics = st.slider("Select Number of Topics", 3, 20, st.session_state["num_topics"], key="num_topics_slider")
            st.session_state["num_topics"] = num_topics # Update session state with the selected number

            # Button to run LDA model training
            if st.button("🚀 Run LDA Model Training"):
                if not corpus:
                    st.error("Cannot run LDA: The corpus is empty. This usually means no meaningful text was found after preprocessing or filtering.")
                else:
                    # The training function is cached, so it runs only if parameters change or first time
                    lda_model = train_gensim_lda_model(corpus, id2word, num_topics)
                    st.session_state["lda_model"] = lda_model
                    st.success("LDA model training complete!")

                    st.subheader("🔑 Top Words per Topic (from LDA Model)")
                    # Display top words for each topic
                    for idx, topic in lda_model.show_topics(num_topics=num_topics, num_words=30, formatted=False):
                        words = ", ".join([w for w, p in topic])
                        st.write(f"**Topic {idx+1}:** {words}")
                        # Initialize topic label if not already set for a new topic
                        if idx not in st.session_state["topic_labels"]:
                            st.session_state["topic_labels"][idx] = f"Topic {idx+1}"
                    
                    # --- MODIFIED SECTION FOR pprint DISPLAY ---
                    st.subheader("Raw LDA Topics for Expert Review")
                    
                    # Capture the output of pprint to a string
                    f = io.StringIO()
                    with redirect_stdout(f):
                        pprint(lda_model.print_topics(num_topics=num_topics, num_words=30)) # Ensure num_topics and num_words match
                    s = f.getvalue()
                    
                    st.code(s) # Display the captured string using st.code
                    # --- END OF MODIFIED SECTION ---

            # Section for assigning custom labels and performing further analysis, only if LDA model is trained
            if st.session_state["lda_model"]:
                st.markdown("---")
                st.subheader("📝 Assign Custom Labels to Topics")

                # Form for manual topic labeling inputs
                with st.form("topic_label_form"):
                    for i in range(num_topics):
                        # Retrieve current label from session state or use default 'Topic X'
                        current_label = st.session_state["topic_labels"].get(i, f"Topic {i+1}")
                        new_label = st.text_input(f"Label for Topic {i+1} (e.g., 'Student Support', 'Course Content')",
                                                 value=current_label, key=f"topic_input_{i}")
                        st.session_state["topic_labels"][i] = new_label # Update label in session state
                    
                    submit_labels = st.form_submit_button("✔️ Apply Topic Labels and Analyze")

                if submit_labels:
                    # Create a copy of the sentiment DataFrame for adding topic assignments
                    df_with_topics = st.session_state["sentiment_df"].copy()
                    
                    # Initialize a list to store assigned topic labels for each original comment
                    # This list will be the same length as the original sentiment_df, ensuring alignment.
                    full_topic_assignments = ["Unassigned"] * len(df_with_topics)

                    # Iterate through each row of the original DataFrame to assign topics
                    for idx, row in df_with_topics.iterrows():
                        cleaned_text_doc = row["Cleaned"]
                        
                        # Handle cases where cleaned text is empty or NaN after preprocessing
                        if pd.isna(cleaned_text_doc) or not cleaned_text_doc.strip():
                            full_topic_assignments[idx] = "Unassigned"
                            continue # Skip to the next document

                        # Convert cleaned text to tokens using simple_preprocess
                        doc_tokens = simple_preprocess(cleaned_text_doc)
                        
                        if not doc_tokens:
                            full_topic_assignments[idx] = "Unassigned"
                            continue # Skip if no tokens are left after simple_preprocess

                        # Convert tokens to Bag-of-Words using the model's dictionary (id2word)
                        # This step is critical for aligning document words to the model's vocabulary.
                        bow_for_this_doc = st.session_state["id2word"].doc2bow(doc_tokens)
                        
                        if not bow_for_this_doc:
                            full_topic_assignments[idx] = "Unassigned"
                            continue # Skip if BOW is empty (e.g., all words filtered out by id2word.filter_extremes)

                        # Get topic probabilities for the current document from the trained LDA model
                        try:
                            topic_probs = st.session_state["lda_model"].get_document_topics(bow_for_this_doc)
                        except IndexError:
                            # Catch potential IndexError if a BOW somehow still contains out-of-vocabulary IDs
                            # This is a safeguard, as previous filters should prevent most cases.
                            st.warning(f"Could not assign topic to document at index {idx} due to vocabulary mismatch. Assigning 'Unassigned'.")
                            full_topic_assignments[idx] = "Unassigned"
                            continue


                        if topic_probs:
                            # Find the topic with the highest probability
                            assigned_topic_id = max(topic_probs, key=lambda x: x[1])[0]
                            # Get the custom label for this topic, or use a default if not set
                            label = st.session_state["topic_labels"].get(assigned_topic_id, f"Topic {assigned_topic_id+1}")
                            full_topic_assignments[idx] = label
                        else:
                            full_topic_assignments[idx] = "Unassigned" # Assign 'Unassigned' if no topic probabilities are found

                    # Add the 'Topic' column to the DataFrame
                    df_with_topics["Topic"] = full_topic_assignments

                    # Filter out 'Unassigned' topics for visualization purposes, as they don't contribute to topic-specific insights
                    df_for_viz = df_with_topics[df_with_topics["Topic"] != "Unassigned"].copy()
                    
                    if not df_for_viz.empty:
                        # --- Topic Analysis Count Plot ---
                        st.subheader("📊 Topic Analysis: Document Counts per Topic")
                        plt.figure(figsize=(12, 7))
                        # Use value_counts().index to ensure bars are ordered by frequency
                        sns.countplot(data=df_for_viz, x="Topic", palette="viridis", order=df_for_viz['Topic'].value_counts().index)
                        plt.title('Distribution of Documents Across Topics')
                        plt.xlabel('Topic Labels')
                        plt.ylabel('Number of Documents')
                        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
                        plt.tight_layout() # Adjust layout to prevent labels from overlapping
                        st.pyplot(plt.gcf()) # Display the plot in Streamlit
                        plt.close() # Close the figure to free memory

                        # --- Topic Polarity Distribution Stacked Bar Chart ---
                        st.subheader("📊 Topic Polarity Distribution")
                        # Group by Topic and Sentiment, then unstack and normalize to percentages
                        df_topic_polarity = df_for_viz.groupby('Topic')['Sentiment'].value_counts(normalize=True).unstack(fill_value=0)
                        
                        # Ensure all three sentiment columns exist and are in a consistent order for plotting
                        sentiment_order = ["😠 Negative", "😐 Neutral", "😊 Positive"]
                        for s_type in sentiment_order:
                            if s_type not in df_topic_polarity.columns:
                                df_topic_polarity[s_type] = 0.0
                        df_topic_polarity = df_topic_polarity[sentiment_order] * 100 # Convert to percentage for display

                        # Define color mapping for sentiments
                        color_mapping = {"😠 Negative": 'red', "😐 Neutral": 'orange', "😊 Positive": 'green'}
                        colors = [color_mapping[col] for col in df_topic_polarity.columns]

                        ax = df_topic_polarity.plot(kind='bar', color=colors, stacked=True, figsize=(12, 7))
                        ax.set_xlabel('Topic')
                        ax.set_ylabel('Percentage of Polarity (%)')
                        ax.set_title('Topic Polarity Distribution')
                        ax.set_xticklabels(df_topic_polarity.index, rotation=45, ha='right')
                        ax.legend(title='Polarity', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside to prevent overlap
                        plt.tight_layout()
                        st.pyplot(ax.get_figure()) # Display the plot
                        plt.close() # Close the figure

# --- Topic Relationship Graph (Based on Sentiment Similarity using Euclidean Distance transformed to Similarity) ---
                        st.subheader("🔗 Topic Relationship Graph (Based on Sentiment Profiles)")
                        st.info("This graph visualizes the relationships between topics. The thickness of a connection indicates the strength of the similarity in their sentiment profiles.")

                        df_sim_ready = df_topic_polarity.fillna(0) # Ensure no NaNs before similarity calculation
                        
                        if df_sim_ready.empty:
                            st.warning("Cannot generate topic relationship graph: No data for sentiment similarity calculation after filtering.")
                        elif len(df_sim_ready) < 2:
                            st.info("Not enough topics (at least 2 required) to draw a relationship graph.")
                        else:
                            topic_names = df_sim_ready.index.tolist()
                            
                            # Calculate similarities using Euclidean distance and convert to a similarity score
                            # A smaller Euclidean distance means higher similarity.
                            # We convert distance to similarity using 1 / (1 + distance) to get values between 0 and 1.
                            topic_polarity_matrix = np.zeros((len(topic_names), len(topic_names)))
                            
                            for i in range(len(topic_names)):
                                for j in range(i + 1, len(topic_names)): # Iterate over unique pairs (i < j)
                                    # Ensure both profiles are 1D arrays for euclidean distance
                                    profile_i = df_sim_ready.iloc[i].values.flatten()
                                    profile_j = df_sim_ready.iloc[j].values.flatten()

                                    dist = euclidean(profile_i, profile_j)
                                    similarity = 1 / (1 + dist) # Transform distance to similarity (0 to 1)
                                    topic_polarity_matrix[i][j] = similarity
                                    topic_polarity_matrix[j][i] = similarity # Make matrix symmetric

                            # --- REWRITTEN NETWORKX GRAPH CODE ---
                            G = nx.Graph()
                            G.add_nodes_from(topic_names)

                            # Add edges for ALL unique pairs, regardless of a hard threshold
                            for i in range(len(topic_polarity_matrix)):
                                for j in range(i + 1, len(topic_polarity_matrix[0])): # Ensure i < j
                                    # ALWAYS add the edge, storing its calculated similarity as 'weight'
                                    G.add_edge(topic_names[i], topic_names[j], weight=topic_polarity_matrix[i][j])
                            
                            # Set the layout of the nodes
                            pos = nx.spring_layout(G)
                            plt.figure(figsize=(12, 8)) # Set figure size for better visualization

                            # Prepare dynamic edge widths based on their weights (similarity scores)
                            edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
                            
                            # Normalize weights to a desired visual range for line thickness (e.g., from 0.5 to 5.0)
                            min_line_width = 0.5
                            max_line_width = 5.0

                            if edge_weights: # Only proceed if there are edges
                                # Handle case where all weights might be identical (avoid division by zero)
                                if max(edge_weights) == min(edge_weights):
                                    # If all weights are the same, assign a default width
                                    widths = [min_line_width] * len(edge_weights)
                                else:
                                    # Normalize weights to the desired min_line_width to max_line_width range
                                    widths = [min_line_width + (w - min(edge_weights)) / (max(edge_weights) - min(edge_weights)) * (max_line_width - min_line_width) for w in edge_weights]
                            else: # Fallback if for some reason no edges were added (shouldn't happen with the logic above unless <2 topics)
                                widths = []

                            # Draw the graph
                            nx.draw(G, pos, with_labels=True, font_weight='bold',
                                    node_color='skyblue', node_size=2000, edge_color='gray',
                                    width=widths, alpha=0.8, font_size=10) 

                            # Set the edge labels (similarity scores)
                            edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
                            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.3) 

                            plt.title('Topic Similarity Graph (Based on Sentiment Profiles)')
                            plt.axis('off') # Hide axes for a cleaner look
                            plt.tight_layout()
                            st.pyplot(plt.gcf()) # Use st.pyplot for Streamlit
                            plt.close() # Close the plot to free memory
                            # --- END OF REWRITTEN NETWORKX GRAPH CODE ---              
                        # --- Interactive LDA Visualization (pyLDAvis) ---
                        st.subheader("📈 Interactive LDA Visualization (pyLDAvis)")
                        st.info("This interactive visualization helps explore topics by showing their relationships and the most relevant terms. Move the mouse over topics and words for details.")
                        
                        # Check if LDA model and its components are available before preparing visualization
                        if st.session_state["lda_model"] and st.session_state["corpus"] and st.session_state["id2word"]:
                            try:
                                # pyLDAvis preparation: mds='mmds' or 'tsne' are common options for dimensionality reduction
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
