import re
from collections import defaultdict
from typing import Dict, List
#from pysentimiento import create_analyzer 
from transformers import pipeline

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"



def chunk_lyrics_to_sentences(lyrics):
    """
    Chunks lyrics into sentences from a SINGLE CONTINUOUS STRING,
    with sentences starting at capitalized words (except 'I') and ending *before* the next.
    """
    lyrics_no_signals = re.sub(r'\[.*?\]', '', lyrics)
    words = lyrics_no_signals.strip().split()

    sentences = []
    current_sentence_words = []
    expecting_start = True

    for word in words:
        if not word:
            continue

        word_capitalized = word and word[0].isupper()
        is_letter_i = word.upper() == 'I' # Check if the word is 'I' (case-insensitive)

        if expecting_start:
            if word_capitalized:
                if current_sentence_words:
                    sentences.append(" ".join(current_sentence_words).strip())
                    current_sentence_words = []
                current_sentence_words.append(word)
                expecting_start = False

        else: # Expecting words within sentence
            if word_capitalized and not is_letter_i: # Capitalized, but NOT 'I' - end sentence before
                sentences.append(" ".join(current_sentence_words).strip())
                current_sentence_words = [word]
                expecting_start = False
            else: # Lowercase word OR Capitalized 'I' - part of current sentence
                current_sentence_words.append(word)

    if current_sentence_words:
        sentences.append(" ".join(current_sentence_words).strip())

    return sentences
#def chunk_lyrics_to_sentences(lyrics):
#    return re.split(r'\[.*?\]', lyrics)

def predict_lyric(data):
    """Returns AnalyzerOutput with emotion probabilities"""
    classifier = pipeline(
        "text-classification", model="tabularisai/multilingual-sentiment-analysis"
    )

    predictions = classifier(data)
    #print(predictions) # Keep print for debugging if needed

    # Transform predictions to the format expected by analyze_song_emotion (.probas)
    probas_dict = {}
    if predictions and predictions[0]: # Ensure predictions and the first element exist
        for prediction in predictions[0]: # predictions is a list of lists, we take the first list
            label = prediction['label']
            score = prediction['score']
            probas_dict[label] = score
    return probas_dict # Transformed dictionary


def analyze_song_emotion(lyrics: str) -> Dict:
    """
    Calculates song emotion based on dominant emotion count per sentence, instead of averaging.
    Returns:
        {
            "dominant_emotion": "...", # Song's dominant emotion (based on counts)
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
    sentences = chunk_lyrics_to_sentences(lyrics)
    sentence_dominant_emotions = {} # Store dominant emotion for each sentence
    dominant_emotion_counts = defaultdict(int) # Count occurrences of each dominant emotion

    for i, sentence in enumerate(sentences): # Enumerate to keep track of sentence index
        if not sentence.strip():
            continue

        print('sentence----------------------------------------------')
        print(sentence)
        result = predict_lyric(sentence) # Now predict_lyric returns probas_dict

        # Determine dominant emotion for the sentence
        sentence_dominant_emotion = max(result, key=result.get, default="neutral") if result else "neutral"
        print(f"Sentence Dominant Emotion: {sentence_dominant_emotion}") # Print sentence dominant emotion

        sentence_dominant_emotions[f"sentence_{i+1}"] = sentence_dominant_emotion # Store sentence dominant emotion
        dominant_emotion_counts[sentence_dominant_emotion] += 1 # Increment count for this dominant emotion

    # Determine song dominant emotion based on counts
    song_dominant_emotion = max(dominant_emotion_counts, key=dominant_emotion_counts.get, default="neutral") if dominant_emotion_counts else "neutral"

    return {
        "dominant_emotion": song_dominant_emotion,
        "sentence_dominant_emotions": sentence_dominant_emotions, # Return sentence-level dominant emotions
        "dominant_emotion_counts": dominant_emotion_counts, # Return counts of each dominant emotion
        "sentence_count": len(sentences)
    }


lyrics_example_continuous_string = "[Verse 1] We clawed, we chained, our hearts in vain We jumped, never asking why We kissed, I fell under your spell A love no one could deny  [Pre-Chorus] Don't you ever say I just walked away I will always want you I can't live a lie, running for my life I will always want you  [Chorus] I came in like a wrecking ball I never hit so hard in love All I wanted was to break your walls All you ever did was wreck me Yeah, you, you wreck me  [Verse 2] I put you high up in the sky And now, you're not coming down It slowly turned, you let me burn And now, we're ashes on the ground  [Pre-Chorus] Don't you ever say I just walked away I will always want you I can't live a lie, running for my life I will always want you  [Chorus] I came in like a wrecking ball I never hit so hard in love All I wanted was to break your walls All you ever did was wreck me I came in like a wrecking ball Yeah, I just closed my eyes and swung Left me crashing in a blazing fall All you ever did was wreck me Yeah, you, you wreck me  [Bridge] I never meant to start a war I just wanted you to let me in And instead of using force I guess I should've let you win I never meant to start a war I just wanted you to let me in I guess I should've let you win  [Interlude] Don't you ever say I just walked away I will always want you  [Chorus] I came in like a wrecking ball I never hit so hard in love All I wanted was to break your walls All you ever did was wreck me I came in like a wrecking ball Yeah, I just closed my eyes and swung Left me crashing in a blazing fall All you ever did was wreck me Yeah, you, you wreck me Yeah, you, you wreck me  [Produced by Dr. Luke and Cirkut] [Video by Terry Richardson]" # No \n characters
#lyrics_example_continuous_string = "People in the world is really worried because of Coronavirus"
analysis = analyze_song_emotion(lyrics_example_continuous_string)
print(f"Song Dominant Emotion (by count): {analysis['dominant_emotion']}") # Updated print statement
print("\nSentence Dominant Emotions:") # Added section for sentence emotions
for sentence_num, emotion in analysis["sentence_dominant_emotions"].items():
    print(f"{sentence_num}: {emotion}")
print("\nDominant Emotion Counts:") # Added section for emotion counts
for emotion, count in analysis["dominant_emotion_counts"].items():
    print(f"{emotion}: {count}")
print(f"\nTotal Sentences: {analysis['sentence_count']}") # Print total sentences