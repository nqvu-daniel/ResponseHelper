import streamlit as st
import functions.main_function as mf
from transformers import pipeline
import torch
import os
from huggingface_hub import login

# Add Hugging Face authentication
@st.cache_resource
def authenticate_huggingface():
    # Get token from Streamlit secrets or environment variable
    token = st.secrets.get("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        st.error("Please set your Hugging Face token in Streamlit secrets or environment variables")
        st.stop()
    login(token)

# Initialize text generation pipeline
@st.cache_resource
def load_text_generation_pipeline():
    device = get_device()
    # Authenticate before loading model
    authenticate_huggingface()
    return pipeline(
        "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )


# Enable MPS fallback for better compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Unified device selection function
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return "mps"
    print("Using CPU")
    return "cpu"

# Initialize text generation pipeline
@st.cache_resource
def load_text_generation_pipeline():
    device = get_device()
    return pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

# TODO: Add custom prompt template here
def generate_prompt(user_input):
    # Placeholder for custom prompt engineering
    return user_input

# Initialize session state for chat messages if not exists
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{
        "role": "assistant",
        "content": "Hello! I'm your Email Sentiment Analyzer and Assistant. Paste your email content for analysis, or ask me to help you with your email."
    }]

# Set page config
st.set_page_config(
    page_title="Email Sentiment Analyzer",
    layout="wide"
)

# Add title and description
st.title("✉️ Email Sentiment Analyzer & Assistant")
st.caption("🚀 Analyze email sentiment and get help with your emails using AI")

# Create two columns for the layout
left_col, right_col = st.columns([2, 1])

# Left column - Chat interface
with left_col:
    # Display chat messages using Streamlit's native chat interface
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Email input through chat (positioned at bottom)
    if email_content := st.chat_input("Type or paste your email or your request here..."):
        if email_content.strip():
            with st.spinner("Processing..."):
                # Add user's email to chat
                st.session_state.chat_messages.append({"role": "user", "content": email_content})
                st.chat_message("user").write(email_content)

                # Check if this is the first user message after assistant's initial greeting
                is_first_user_message = len([msg for msg in st.session_state.chat_messages if msg["role"] == "user"]) == 1

                if is_first_user_message:
                    # First message is treated as email for sentiment analysis
                    analysis = mf.analyze_email_sentiment(email_content)

                    assistant_message = f"📊 Sentiment Analysis of your email:\n\n"
                    assistant_message += f"Overall Sentiment: **{analysis['dominant_emotion'].upper()}**\n\n"
                    assistant_message += "Sentence-by-Sentence Analysis:\n"

                    # Split email by both sentences and newlines
                    raw_sentences = mf.chunk_email_to_sentences(email_content)
                    sentences = []
                    for sentence in raw_sentences:
                        # Split further by newlines and filter out empty lines
                        split_lines = [line.strip() for line in sentence.split('\n') if line.strip()]
                        sentences.extend(split_lines)

                    # Analyze each segment
                    for i, sentence in enumerate(sentences):
                        sentiment_result = mf.predict_sentiment(sentence)
                        dominant_emotion = max(sentiment_result.items(), key=lambda x: x[1], default=("neutral", 0))
                        emotion, score = dominant_emotion
                        assistant_message += f"\n📝 Segment {i+1}:\n"
                        assistant_message += f"> {sentence}\n"
                        assistant_message += f"Sentiment: **{emotion.upper()}** (Confidence: {score:.1%})\n"
                else:
                    # Subsequent messages are treated as commands/requests related to the *previous* email
                    text_generator = load_text_generation_pipeline()

                    # Get the previous email content
                    previous_email_content = ""
                    user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
                    if len(user_messages) > 1:
                        previous_email_content = user_messages[-2]['content']

                    # **IMPROVED PROMPT:**
                    messages = [
                        {
                            "role": "user", 
                            "content": f"You are an AI email assistant. Help with this email request:\n\nPrevious email:\n{previous_email_content}\n\nUser request:\n{email_content}"
                        }
                    ]

                    try:
                        # Generate response with Gemma specific parameters
                        response = text_generator(
                            messages,
                            max_new_tokens=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )

                        assistant_message = "I am sorry, I could not generate a relevant email response. Please try again."
                        if response:
                            # Extract the generated text from Gemma's response
                            generated_text = response[0]["generated_text"][-1]["content"].strip()
                            assistant_message = generated_text

                            if len(assistant_message) < 10 or "I am sorry" in assistant_message:
                                assistant_message = "I can help with emails, but I need a clearer request. Could you please be more specific about what you'd like me to do with the email?"

                    except Exception as e:
                        assistant_message = f"Sorry, there was an error generating the email response. Please try again. Error details: {e}"
                # Add analysis/response to chat
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message})
                st.chat_message("assistant").write(assistant_message)

# Right column - Settings and Info
with right_col:
    st.subheader("About")
    st.info("""
    This tool analyzes the emotional tone of your emails and can act as your email assistant using AI.

    How to use:
    1. **First Message:** Type or paste your email in the chat to analyze its sentiment.
    2. **Subsequent Messages:** Ask me to help with your email, like "fix this email", "rewrite it", "respond to it", etc.
    3. I will use the previous email as context to understand your requests.
    """)

    # Add a clear chat button
    if st.button("Clear Chat History", type="secondary", key="clear_chat_settings"):
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": "Hello! I'm your Email Sentiment Analyzer and Assistant. Paste your email content for analysis, or ask me to help you with your email."
        }]
        st.rerun()

# Right column - Analysis visualization
with right_col:
    st.subheader("Sentiment Analysis Results")

    # Only show analysis if there are messages
    if st.session_state.chat_messages:
        # Get the first analysis message
        first_analysis = None
        for message in st.session_state.chat_messages:
            if message["role"] == "assistant" and "📊 Sentiment Analysis of your email:" in message["content"]:
                first_analysis = message["content"]
                break

        if first_analysis:
            st.info(first_analysis)

    else:
        st.info("Enter an email on the left panel to see the sentiment analysis results here.")