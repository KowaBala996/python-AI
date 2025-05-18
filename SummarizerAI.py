import streamlit as st
from transformers import pipeline

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="AI Text Summarizer", layout="centered")

# Load the summarizer model
@st.cache_resource
def load_model():
    """Load and cache the summarization model."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize the summarizer
summarizer = load_model()

# Streamlit app layout
st.title("🧠 AI Text Summarizer")
st.write("Paste any long text or article and get a smart summary.")

# Input text area
text = st.text_area("Enter your text here:", height=300)

# Summarization logic
def summarize_text(input_text):
    """Generate a summary for the given input text."""
    return summarizer(input_text, max_length=150, min_length=100, do_sample=False)

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Summarizing..."):
            try:
                summary = summarize_text(text)
                st.success("Summary:")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
