# Email Sentiment Analyzer & Assistant

An application that analyzes email sentiment and provides AI-powered email assistance

Another fun little project I made! :3

## Features

- 📊 Email sentiment analysis
- 💬 AI-powered email assistance (rewriting, responding, fixing)
- 📱 User-friendly chat interface
- 📈 Detailed sentence-by-sentence sentiment breakdown

## Setup

1. Clone the repository
2. Install dependencies: (i mightve missed a few here)
   ```bash
   pip install streamlit transformers torch huggingface_hub seaborn
   ```
3. Set up Hugging Face authentication:
   - Get your Hugging Face token from https://huggingface.co/settings/tokens
   - Set it as an environment variable: `HUGGINGFACE_TOKEN=your_token`
   - Or add it to Streamlit secrets

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Paste your email in the chat to analyze its sentiment
3. Use follow-up commands like "fix this email" or "make it more professional" for AI assistance

## Dependencies

- Streamlit for the web interface
- HuggingFace Transformers for sentiment analysis and text generation
- Google's Gemma-2-2b for email assistance
