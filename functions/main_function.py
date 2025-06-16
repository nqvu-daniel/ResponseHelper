import re
from collections import defaultdict
from typing import Dict, List
from transformers import pipeline
import torch
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def get_device():
    """Determine the best available device for computation"""
    if torch.cuda.is_available():
        print("Using CUDA")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return "mps"
    print("Using CPU")
    return "cpu"

def predict_sentiment(data):
    """Returns sentiment analysis with probabilities"""
    device = get_device()
    classifier = pipeline(
        "text-classification",
        model="clapAI/modernBERT-base-multilingual-sentiment",
        #model="./fine_turned_model"
        #"nlptown/bert-base-multilingual-uncased-sentiment",
        #"tabularisai/multilingual-sentiment-analysis",
        device=device
    )
    
    predictions = classifier(data)
    print(predictions)
    probas_dict = {}
    if isinstance(predictions, list) and predictions:
        prediction = predictions[0]
        if isinstance(prediction, dict):
            label = prediction['label']
            score = prediction['score']
            probas_dict[label] = score
    print(probas_dict)            
    return probas_dict

def chunk_email_to_sentences(email_content):
    """
    Splits email content into proper sentences using punctuation and natural breaks.
    """
    # Clean the content (remove excessive whitespace and normalize line endings)
    clean_content = re.sub(r'\s+', ' ', email_content).strip()
    
    # Split into sentences using common sentence endings (.!?) followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_content)
    return [s.strip() for s in sentences if s.strip()]

def analyze_email_sentiment(email_content: str) -> Dict:
    """
    Analyzes email sentiment based on dominant emotion count per sentence.
    Returns:
        {
            "dominant_emotion": "...", # Email's dominant emotion (based on counts)
            "sentence_dominant_emotions": { # Dominant emotion for each sentence
                "sentence_1": "...",
                "sentence_2": "...",
                ...
            },
            "dominant_emotion_counts": { # Count of sentences with each dominant emotion
                "emotion_1": count_1,
                "emotion_2": count_2,
                ...
            },
            "sentence_count": total_sentences
        }
    """
    sentences = chunk_email_to_sentences(email_content)
    sentence_dominant_emotions = {}
    dominant_emotion_counts = defaultdict(int)

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        result = predict_sentiment(sentence)

        # Determine dominant emotion for the sentence
        sentence_dominant_emotion = max(result, key=result.get, default="neutral") if result else "neutral"
        sentence_dominant_emotions[f"sentence_{i+1}"] = sentence_dominant_emotion
        dominant_emotion_counts[sentence_dominant_emotion] += 1

    # Determine email dominant emotion based on counts
    email_dominant_emotion = max(dominant_emotion_counts, key=dominant_emotion_counts.get, default="neutral") if dominant_emotion_counts else "neutral"

    return {
        "dominant_emotion": email_dominant_emotion,
        "sentence_dominant_emotions": sentence_dominant_emotions,
        "dominant_emotion_counts": dominant_emotion_counts,
        "sentence_count": len(sentences)
    }

if __name__ == "__main__":
    email_example = """
    Dear Team,
    I am extremely disappointed with the latest project delivery. The quality is below our standards and the deadline was missed by two weeks. 
    I expect immediate action to rectify these issues. Please provide a detailed plan by tomorrow.
    Regards,
    John
    """
    
    analysis = analyze_email_sentiment(email_example)
    print(f"Email Dominant Emotion: {analysis['dominant_emotion']}")
    print("\nSentence Dominant Emotions:")
    for sentence_num, emotion in analysis["sentence_dominant_emotions"].items():
        print(f"{sentence_num}: {emotion}")
    print("\nDominant Emotion Counts:")
    for emotion, count in analysis["dominant_emotion_counts"].items():
        print(f"{emotion}: {count}")
    print(f"\nTotal Sentences: {analysis['sentence_count']}")