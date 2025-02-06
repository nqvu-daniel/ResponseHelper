import streamlit as st
import functions.main_function as mf
from transformers import pipeline
import torch
import os

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
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    return pipeline("text-generation", model=model_name, device=device)

# TODO: Add custom prompt template here
def generate_prompt(user_input):
    # Placeholder for custom prompt engineering
    return user_input

# Initialize session state for chat messages if not exists
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{
        "role": "assistant",
        "content": "Hello! I'm your Email Sentiment Analyzer. Paste your email content, and I'll analyze its emotional tone."
    }]

# Set page config
st.set_page_config(
    page_title="Email Sentiment Analyzer",
    layout="wide"
)

# Add title and description
st.title("✉️ Email Sentiment Analyzer")
st.caption("🚀 Analyze the emotional tone of your emails with AI")

# Create two columns for the layout
left_col, right_col = st.columns([2, 1])

# Left column - Chat interface
with left_col:
    # Display chat messages using Streamlit's native chat interface
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Email input through chat (positioned at bottom)
    if email_content := st.chat_input("Type or paste your email here..."):
        if email_content.strip():
            with st.spinner("Processing..."):
                # Add user's email to chat
                st.session_state.chat_messages.append({"role": "user", "content": email_content})
                st.chat_message("user").write(email_content)
                
                # Get sentiment analysis
                analysis = mf.analyze_email_sentiment(email_content)
                
                # Check if this is the first message
                is_first_message = len([msg for msg in st.session_state.chat_messages if msg["role"] == "user"]) == 1
                
                if is_first_message:
                    # Format analysis results for first message
                    assistant_message = f"📊 Here's my analysis of your email:\n\n"
                    assistant_message += f"Overall Sentiment: **{analysis['dominant_emotion'].upper()}**\n\n"
                    assistant_message += "Sentence-by-Sentence Analysis:\n"
                    
                    # Analyze each sentence
                    sentences = mf.chunk_email_to_sentences(email_content)
                    for i, sentence in enumerate(sentences):
                        sentiment_result = mf.predict_sentiment(sentence)
                        dominant_emotion = max(sentiment_result.items(), key=lambda x: x[1], default=("neutral", 0))
                        emotion, score = dominant_emotion
                        assistant_message += f"\n📝 Sentence {i+1}:\n"
                        assistant_message += f"> {sentence}\n"
                        assistant_message += f"Sentiment: **{emotion.upper()}** (Confidence: {score:.1%})\n"
                else:
                    # For subsequent messages, use DeepSeek model with context
                    text_generator = load_text_generation_pipeline()
                    
                    # Create context with email and sentiment analysis
                    context = f"Previous email: {email_content}\n"
                    context += f"Sentiment analysis: {analysis['dominant_emotion']}\n"
                    context += "Based on this context, provide a helpful response.\n"
                    
                    # Generate response using DeepSeek model
                    response = text_generator(
                        context,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7
                    )
                    assistant_message = response[0]['generated_text'] if response else "I apologize, but I couldn't generate a response."
                
                # Add analysis to chat
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message})
                st.chat_message("assistant").write(assistant_message)

# Right column - Settings and Info
with right_col:
    st.subheader("About")
    st.info("""
    This tool analyzes the emotional tone of your emails using advanced AI.
    
    How to use:
    1. Type or paste your email in the chat
    2. The AI will analyze the overall sentiment
    3. Get a detailed breakdown of each sentence
    """)
    
    # Add a clear chat button
    if st.button("Clear Chat History", type="secondary", key="clear_chat_settings"):
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": "Hello! I'm your Email Sentiment Analyzer. Paste your email content, and I'll analyze its emotional tone."
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
            if message["role"] == "assistant" and "📊 Here's my analysis of your email:" in message["content"]:
                first_analysis = message["content"]
                break
        
        if first_analysis:
            st.info(first_analysis)
            
            # Add a clear chat button
            if st.button("Clear Chat History", key="clear_chat_results"):
                st.session_state.chat_messages = []
                st.rerun()
    else:
        st.info("Enter an email on the left panel to see the sentiment analysis results here.")