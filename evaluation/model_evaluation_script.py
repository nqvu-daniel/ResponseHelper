import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import pipeline
import torch
import sys
sys.path.append('../')
from functions.main_function import predict_sentiment, get_device

def load_test_data(test_file, sample_size=100):
    """Load and sample test data"""
    test_data = pd.read_csv(test_file, header=None, names=['rating', 'title', 'text'])
    test_sample = test_data.sample(n=sample_size, random_state=42)
    return test_sample

def map_sentiment_to_rating(sentiment_result):
    """Map sentiment analysis results to numerical ratings"""
    # Return neutral (3) if no result
    if not sentiment_result or not isinstance(sentiment_result, list) or not sentiment_result:
        return 3
    
    # Get the prediction dictionary from the list
    prediction = sentiment_result[0]
    if not isinstance(prediction, dict) or 'label' not in prediction:
        return 3
    
    # Direct mapping from sentiment labels to ratings
    sentiment_map = {
        'Very Negative': 1,
        'Negative': 2,
        'Neutral': 3,
        'Positive': 4,
        'Very Positive': 5
    }
    
    # Get the sentiment label and return corresponding rating
    label = prediction['label']
    return sentiment_map.get(label, 3)  # Default to neutral (3) if label not found

def evaluate_model(test_sample):
    """Evaluate model performance on test data"""
    predictions = []
    true_pred = []
    
    for text in test_sample['text']:
        result = predict_sentiment(text)
        true_pred.append(result)
        predictions.append(map_sentiment_to_rating(result))
    
    predictions = np.array(predictions)
    true_ratings = test_sample['rating'].values
    
    return predictions, true_ratings

def plot_confusion_matrix(true_ratings, predictions):
    """Plot confusion matrix and print classification report"""
    cm = confusion_matrix(true_ratings, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Rating')
    plt.ylabel('True Rating')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print('Classification Report:')
    print(classification_report(true_ratings, predictions))

def analyze_errors(test_sample, predictions, true_ratings):
    """Analyze prediction errors"""
    errors = np.abs(true_ratings - predictions)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=range(6), align='left', rwidth=0.8)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Error (Rating Points)')
    plt.ylabel('Count')
    plt.xticks(range(5))
    plt.savefig('error_distribution.png')
    plt.close()
    
    print(f'Mean Absolute Error: {np.mean(errors):.2f}')
    print(f'Median Absolute Error: {np.median(errors):.2f}')
    
    # Analyze top error cases
    test_sample['predicted_rating'] = predictions
    test_sample['error'] = errors
    
    print('\nTop 5 cases with largest prediction errors:')
    error_cases = test_sample.nlargest(5, 'error')
    for _, case in error_cases.iterrows():
        print(f'True Rating: {case["rating"]}')
        print(f'Predicted Rating: {case["predicted_rating"]}')
        print(f'Text: {case["text"]}\n')

def main():
    # Load and prepare test data
    test_sample = load_test_data('test.csv')
    print(f'Sample size: {len(test_sample)}\n')
    
    # Evaluate model
    predictions, true_ratings = evaluate_model(test_sample)
    
    # Generate and plot confusion matrix
    plot_confusion_matrix(true_ratings, predictions)
    
    # Analyze errors
    analyze_errors(test_sample, predictions, true_ratings)

if __name__ == "__main__":
    main()