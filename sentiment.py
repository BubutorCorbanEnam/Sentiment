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

from PIL import Image
import io

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
nltk.download('punkt_tab')
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
st.set_page_config(page_title="Sentiment Typing App", layout="centered")

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
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="ucc.png" width="150">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="university-header"><h2>University of Cape Coast</h2><p>Sentiment Analysis Web App</p></div>', unsafe_allow_html=True)

# ------------------ PROFESSIONAL BACKGROUND ------------------
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Built by Bubutor Corban Enam after participating in an NLP training session organized by Professor Andy. This app allows users to analyze the sentiment of comments using natural language processing.
    
    It supports both batch analysis via CSV upload and manual typing. Results include polarity, subjectivity, sentiment type, and visual insights. Ideal for researchers, marketers, and educators.
    """)

# ------------------ SESSION STATE ------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Original Comment", "Cleaned Comment", "Polarity", "Subjectivity", "Sentiment", "Opinion/Fact"
    ])

# File upload section
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

                # WordCloud from all cleaned comments
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

# Analysis history
if not st.session_state.results_df.empty:
    st.markdown("### 🗂️ Analysis History")
    st.dataframe(st.session_state.results_df)

    st.download_button(
        "📥 Download All Results",
        data=st.session_state.results_df.to_csv(index=False).encode(),
        file_name="all_sentiment_results.csv",
        mime="text/csv"
    )

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

    st.markdown("### 📌 Polarity vs Subjectivity")
    scatter = alt.Chart(st.session_state.results_df).mark_circle(size=70).encode(
        x='Polarity',
        y='Subjectivity',
        color='Sentiment',
        tooltip=['Original Comment', 'Polarity', 'Subjectivity', 'Sentiment']
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

# Analyze single comment (last step)
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
            #st.markdown(f"**Sentiment:** {sentiment}")
            #st.markdown(f"**Polarity:** `{polarity}`")
            #st.markdown(f"**Subjectivity:** `{subjectivity}`")
            #st.markdown(f"**Type:** {opinion_type}")
            st.markdown(f"📝 Your comment expresses a **{sentiment}** sentiment and is more **{opinion_type.lower()}-based**.")

            # Append to session state
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
