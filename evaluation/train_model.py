import numpy as np
import evaluate
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
import pandas as pd

def load_and_preprocess_data(train_file):
    """Load and preprocess the training data"""
    # Load training data
    df = pd.read_csv(train_file, header=None, names=['rating', 'title', 'text'])
    
    # Convert string ratings to integers
    df['rating'] = df['rating'].astype(int)
    
    # Map ratings directly to 0-2 range for model training (3 classes)
    # Map ratings 1-2 to 0 (negative)
    # Map rating 3 to 1 (neutral)
    # Map ratings 4-5 to 2 (positive)
    df['label'] = df['rating'].apply(lambda x: 0 if x <= 2 else (2 if x >= 4 else 1))
    
    return df

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(train_data, model_name="clapAI/modernBERT-base-multilingual-sentiment"):
    """Initialize and train the model"""
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        torch_dtype="auto"
    )
    
    # Tokenize the text data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    # Tokenize dataset
    train_encodings = tokenize_function({'text': train_data['text'].tolist()})
    
    # Create torch dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, train_data['label'].tolist())
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",  # No evaluation since we'll use separate test set
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

def main():
    # Load and preprocess data
    train_data = load_and_preprocess_data('train.csv')
    
    # Train model
    train_model(train_data)

if __name__ == "__main__":
    main()