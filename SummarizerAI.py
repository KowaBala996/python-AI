import streamlit as st
from transformers import pipeline
import time

# This must be the first Streamlit command
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the summarizer model
@st.cache_resource
def load_model():
    """Load and cache the summarization model."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
max_length = st.sidebar.slider("Maximum summary length", 50, 500, 250)
min_length = st.sidebar.slider("Minimum summary length", 30, 200, 100)
do_sample = st.sidebar.checkbox("Use sampling", False)

if do_sample:
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9)
else:
    temperature = 0.7
    top_p = 0.9

# About section in sidebar
st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "✨ Welcome to the AI Text Summarizer! ✨\n\n"
    "This lovely web creates concise summaries of your text with the power of AI. "
    "Simply paste your content and watch the magic happen! "
    "Enjoy turning your long texts into short, meaningful summaries!"
)

# Main app layout
st.title("🧠AI Text Summarizer ")
st.write("Paste any long text or article and get a smart, concise summary in seconds!")

# Input options
input_method = st.radio(
    "Choose input method:",
    ("Text Input", "Sample Text")
)

if input_method == "Sample Text":
    sample_texts = {
        "Technology Article": """Artificial intelligence has transformed numerous industries, from healthcare to finance, transportation to entertainment. Machine learning models, particularly deep learning architectures, have demonstrated remarkable capabilities in pattern recognition, natural language processing, and decision-making tasks. However, these advancements raise important ethical considerations regarding privacy, bias, and the future of work. As AI systems become more autonomous and integrated into critical infrastructure, ensuring transparency, accountability, and safety becomes paramount. Researchers are actively working on developing interpretable AI, robust evaluation metrics, and governance frameworks to address these challenges. Despite concerns, the potential benefits of AI in solving complex problems like climate change, disease detection, and resource optimization are immense. The key lies in responsible development and deployment that prioritizes human values while harnessing the computational power of these systems.""",
        "Science News": """Scientists have discovered a new species of deep-sea creature living near hydrothermal vents in the Pacific Ocean. This remarkable organism has evolved unique adaptations to thrive in one of Earth's most extreme environments, where temperatures can reach over 400°C and toxic chemicals abound. The creature, tentatively named Thermopodus extremis, possesses specialized proteins that remain stable under high pressure and temperature. Researchers believe studying these adaptations could lead to breakthroughs in biotechnology and pharmaceutical development. The discovery also highlights how much remains unknown about our ocean ecosystems, which cover over 70% of the planet's surface yet remain largely unexplored. Funding for deep-sea exploration has increased in recent years as scientists race to understand these unique habitats before they face disruption from deep-sea mining operations and climate change impacts.""",
        "Business Report": """The global economy faces significant headwinds in the coming quarters, according to analysts at major financial institutions. Rising inflation, supply chain disruptions, and geopolitical tensions have created a challenging environment for businesses and investors alike. Central banks around the world are navigating difficult monetary policy decisions, balancing inflation concerns against risks of economic slowdown. Meanwhile, consumer spending patterns continue to evolve in the post-pandemic landscape, with digital transformation accelerating across sectors. Companies that have successfully adapted their business models show stronger resilience, particularly those embracing sustainability initiatives and flexible work arrangements. Small businesses remain particularly vulnerable to economic volatility, though targeted support programs have helped mitigate some impacts. Economists project moderate growth overall, with significant regional variations depending on vaccination rates, policy responses, and exposure to global trade fluctuations."""
    }
    
    sample_choice = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
    text = sample_texts[sample_choice]
    st.text_area("Sample text:", text, height=300, disabled=True)
else:
    text = st.text_area("Enter your text here:", height=300)

# Text statistics
if text:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", len(text))
    with col2:
        st.metric("Words", len(text.split()))
    with col3:
        st.metric("Sentences", text.count('.') + text.count('!') + text.count('?'))

# Summarization logic
def summarize_text(input_text, max_len=150, min_len=100, use_sampling=False, temp=0.7, top_p_val=0.9):
    """Generate a summary for the given input text."""
    # Initialize the summarizer if not already loaded
    if 'summarizer' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.summarizer = load_model()
    
    summarizer = st.session_state.summarizer
    
    # Check if the input text is too short
    if len(input_text.split()) < min_len:
        return {"error": f"Text is too short for meaningful summarization. Please provide at least {min_len} words."}
    
    # Generate summary
    if use_sampling:
        summary = summarizer(
            input_text, 
            max_length=max_len, 
            min_length=min_len, 
            do_sample=True,
            temperature=temp,
            top_p=top_p_val
        )
    else:
        summary = summarizer(
            input_text, 
            max_length=max_len, 
            min_length=min_len, 
            do_sample=False
        )
    
    return summary[0]

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Summarizing..."):
            try:
                # Add artificial delay to show the spinner
                time.sleep(0.5)
                
                result = summarize_text(
                    text, 
                    max_len=max_length, 
                    min_len=min_length,
                    use_sampling=do_sample,
                    temp=temperature,
                    top_p_val=top_p
                )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("✨ Your lovely summary is ready! ✨")
                    st.write(result['summary_text'])
                    
                    # Comparison metrics
                    st.subheader("Compression Stats")
                    original_words = len(text.split())
                    summary_words = len(result['summary_text'].split())
                    compression_ratio = (original_words - summary_words) / original_words * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Words", original_words)
                    with col2:
                        st.metric("Summary Words", summary_words)
                    with col3:
                        st.metric("Compression", f"{compression_ratio:.1f}%")
                    
                    # Download option
                    st.download_button(
                        label="Download Summary",
                        data=result['summary_text'],
                        file_name="summary.txt",
                        mime="text/plain"
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Try with a shorter text or adjust the parameters in the sidebar.")

# Add instructions
with st.expander("How to use this app"):
    st.markdown("""
    1. **Enter your text** in the text area or select a sample text
    2. **Adjust parameters** in the sidebar if needed:
       - Maximum/minimum summary length
       - Sampling options for more creative summaries
    3. **Click 'Summarize'** to generate the summary
    4. **Download** your summary if desired
    
    For best results, use texts with at least 200 words.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using AI technology")
st.markdown("✨ Thank you for using our text summarizer! Have a wonderful day! ✨")