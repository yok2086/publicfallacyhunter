import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import random
import os
warnings.filterwarnings('ignore')

class FallacyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FallacyHunterSSL:
    def __init__(self, model_name='distilbert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def prepare_data(self, df):
        """Prepare data and create label mappings"""
        print("Preparing data and creating label mappings...")

        # ADD THESE 2 LINES:
        df = df.dropna(subset=['suspected_fallacy'])  # Remove rows with NaN fallacies
        df = df[df['suspected_fallacy'] != 'nan']     # Remove 'nan' string fallacies
        
        # Create label mappings
        unique_fallacies = df['suspected_fallacy'].unique()
        self.label_to_id = {fallacy: idx for idx, fallacy in enumerate(unique_fallacies)}
        self.id_to_label = {idx: fallacy for fallacy, idx in self.label_to_id.items()}
        
        print(f"Found {len(unique_fallacies)} fallacy types:")
        for fallacy, idx in self.label_to_id.items():
            count = len(df[df['suspected_fallacy'] == fallacy])
            print(f"  {idx}: {fallacy} ({count} examples)")
        
        # Convert labels to IDs
        df['label_id'] = df['suspected_fallacy'].map(self.label_to_id)
        
        return df
    
    def initialize_model(self, num_labels):
        """Initialize the transformer model"""
        print(f"Initializing {self.model_name} model with {num_labels} labels...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        
    def create_datasets(self, train_texts, train_labels, val_texts, val_labels):
        """Create training and validation datasets"""
        train_dataset = FallacyDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = FallacyDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, output_dir='./fallacy_model', 
                   num_epochs=6, batch_size=8, learning_rate=3e-5):
        """Train the model with given datasets"""
        print(f"Training model for {num_epochs} epochs...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        return trainer
    
    def predict_with_confidence(self, texts, batch_size=16):
        """Predict fallacies with confidence scores"""
        self.model.eval()
        predictions = []
        confidences = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Get predicted classes and confidence scores
                max_probs, predicted_classes = torch.max(probs, dim=-1)
                
                predictions.extend(predicted_classes.cpu().numpy())
                confidences.extend(max_probs.cpu().numpy())
        
        return predictions, confidences
    
    def find_uncertain_examples(self, texts, uncertainty_threshold=0.7):
        """Find examples where the model is most uncertain - good candidates for manual annotation"""
        predictions, confidences = self.predict_with_confidence(texts)
        
        # Find examples with medium confidence (uncertain but not random)
        uncertain_mask = (np.array(confidences) >= 0.4) & (np.array(confidences) <= uncertainty_threshold)
        uncertain_texts = [text for i, text in enumerate(texts) if uncertain_mask[i]]
        uncertain_confidences = [conf for i, conf in enumerate(confidences) if uncertain_mask[i]]
        uncertain_predictions = [pred for i, pred in enumerate(predictions) if uncertain_mask[i]]
        
        # Sort by confidence (lowest first - most uncertain)
        sorted_indices = np.argsort(uncertain_confidences)
        
        results = []
        for idx in sorted_indices[:50]:  # Top 50 most uncertain
            results.append({
                'text': uncertain_texts[idx],
                'predicted_fallacy': self.id_to_label[uncertain_predictions[idx]],
                'confidence': uncertain_confidences[idx]
            })
        
        return results
    
    def pseudo_label_unlabeled_data(self, unlabeled_texts, confidence_threshold=0.8):
        """Generate pseudo-labels for unlabeled data"""
        print(f"Generating pseudo-labels with confidence threshold {confidence_threshold}...")
        
        predictions, confidences = self.predict_with_confidence(unlabeled_texts)
        
        # Filter by confidence threshold
        high_confidence_mask = np.array(confidences) >= confidence_threshold
        pseudo_labeled_texts = [text for i, text in enumerate(unlabeled_texts) if high_confidence_mask[i]]
        pseudo_labels = [pred for i, pred in enumerate(predictions) if high_confidence_mask[i]]
        pseudo_confidences = [conf for i, conf in enumerate(confidences) if high_confidence_mask[i]]
        
        print(f"Generated {len(pseudo_labeled_texts)} pseudo-labels from {len(unlabeled_texts)} texts")
        if len(pseudo_confidences) > 0:
            print(f"Average confidence: {np.mean(pseudo_confidences):.3f}")
        
        # Show distribution of pseudo-labels
        if len(pseudo_labels) > 0:
            pseudo_label_counts = Counter(pseudo_labels)
            print("Pseudo-label distribution:")
            for label_id, count in pseudo_label_counts.items():
                fallacy_name = self.id_to_label[label_id]
                print(f"  {fallacy_name}: {count}")
        
        return pseudo_labeled_texts, pseudo_labels, pseudo_confidences
    
    def semi_supervised_training(self, labeled_df, unlabeled_texts, 
                               confidence_threshold=0.8, iterations=3):
        """Main semi-supervised learning pipeline"""
        print("Starting Semi-Supervised Learning Pipeline")
        print("=" * 50)
        
        # Prepare labeled data
        labeled_df = self.prepare_data(labeled_df)
        
        # Initialize model
        num_labels = len(self.label_to_id)
        self.initialize_model(num_labels)
        
        # Initial split of labeled data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            labeled_df['comment_text'].tolist(),
            labeled_df['label_id'].tolist(),
            test_size=0.2,
            stratify=labeled_df['label_id'],
            random_state=42
        )
        
        print(f"Initial training set: {len(train_texts)} examples")
        print(f"Validation set: {len(val_texts)} examples")
        print(f"Unlabeled set: {len(unlabeled_texts)} examples")
        
        # Keep track of training history
        training_history = []
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
            
            # Create datasets
            train_dataset, val_dataset = self.create_datasets(
                train_texts, train_labels, val_texts, val_labels
            )
            
            # Train model
            trainer = self.train_model(train_dataset, val_dataset)
            
            # Evaluate on validation set
            eval_results = trainer.evaluate()
            print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
            
            # Store training history
            training_history.append({
                'iteration': iteration + 1,
                'train_size': len(train_texts),
                'val_loss': eval_results['eval_loss']
            })
            
            # Generate pseudo-labels (except for last iteration)
            if iteration < iterations - 1 and len(unlabeled_texts) > 0:
                pseudo_texts, pseudo_labels, pseudo_confidences = self.pseudo_label_unlabeled_data(
                    unlabeled_texts, confidence_threshold
                )
                
                if len(pseudo_texts) > 0:
                    # Add pseudo-labeled data to training set
                    train_texts.extend(pseudo_texts)
                    train_labels.extend(pseudo_labels)
                    
                    print(f"Added {len(pseudo_texts)} pseudo-labeled examples to training set")
                    print(f"New training set size: {len(train_texts)}")
                else:
                    print("No high-confidence pseudo-labels generated. Stopping early.")
                    break
        
        return trainer, training_history
    
    def evaluate_model(self, test_texts, test_labels):
        """Evaluate the trained model"""
        print("Evaluating final model...")
        
        predictions, confidences = self.predict_with_confidence(test_texts)
        
        # Generate classification report
        report = classification_report(
            test_labels, predictions,
            target_names=[self.id_to_label[i] for i in range(len(self.id_to_label))],
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(
            test_labels, predictions,
            target_names=[self.id_to_label[i] for i in range(len(self.id_to_label))]
        ))
        
        return report, predictions, confidences
    
    def plot_training_history(self, training_history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        iterations = [h['iteration'] for h in training_history]
        train_sizes = [h['train_size'] for h in training_history]
        plt.plot(iterations, train_sizes, 'b-o')
        plt.title('Training Set Size Growth')
        plt.xlabel('Iteration')
        plt.ylabel('Training Set Size')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        val_losses = [h['val_loss'] for h in training_history]
        plt.plot(iterations, val_losses, 'r-o')
        plt.title('Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Validation Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path='./fallacy_hunter_model'):
        """Save the trained model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save label mappings
        import json
        with open(f'{path}/label_mappings.json', 'w') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path='./fallacy_hunter_model'):
        """Load a saved model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load label mappings
        import json
        with open(f'{path}/label_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.label_to_id = mappings['label_to_id']
            self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
        
        self.model.to(self.device)
        print(f"Model loaded from {path}")

def filter_quality_comments(texts, min_length=30, max_length=800):
    """Filter comments by length and basic quality"""
    filtered = []
    for text in texts:
        if (min_length <= len(text) <= max_length and 
            len(text.split()) >= 5 and  # At least 5 words
            text.count('.') <= 10):     # Not too fragmented
            filtered.append(text)
    return filtered

def main():
    """
    STEP 1: PREPARE YOUR DATA FILES
    Replace these file paths with your actual file paths:
    """
    
    # annotated data file
    ANNOTATED_FILE = 'annotation_batch_augmented.csv'  
    
    # unlabeled comments file 
    UNLABELED_FILE = 'filtered_comments_for_annotation.csv'  
    
    print("=" * 60)
    print("FALLACY HUNTER SEMI-SUPERVISED LEARNING PIPELINE")
    print("=" * 60)
    
    # Initialize the SSL pipeline
    fallacy_hunter = FallacyHunterSSL(model_name='distilbert-base-uncased')
    
    """
    STEP 2: LOAD ANNOTATED DATA
    """
    print("\nLoading annotated data...")
    df = pd.read_csv(ANNOTATED_FILE)
    print(f"Loaded {len(df)} annotated examples")
    print(f"Fallacy types: {df['suspected_fallacy'].nunique()}")
    print("\nFallacy distribution:")
    print(df['suspected_fallacy'].value_counts())
    
    """
    STEP 3: LOAD AND PREPARE UNLABELED DATA
    """
    print(f"\nLoading unlabeled data...")
    unlabeled_df = pd.read_csv(UNLABELED_FILE)
    all_unlabeled_texts = unlabeled_df['comment_text'].tolist()
    print(f"Loaded {len(all_unlabeled_texts)} unlabeled comments")
    
    # Sample for optimal performance (3000 is good balance)
    random.seed(42)  # For reproducibility
    target_size = min(3000, len(all_unlabeled_texts))
    unlabeled_texts = random.sample(all_unlabeled_texts, target_size)
    print(f"📝 Sampled {len(unlabeled_texts)} comments for training")
    
    # Quality filtering
    unlabeled_texts = filter_quality_comments(unlabeled_texts)
    print(f"🔍 After quality filtering: {len(unlabeled_texts)} comments")
    
    """
    STEP 4: RUN SEMI-SUPERVISED TRAINING
    """
    print(f"\n🚀 Starting semi-supervised training...")
    trainer, history = fallacy_hunter.semi_supervised_training(
        labeled_df=df,
        unlabeled_texts=unlabeled_texts,
        confidence_threshold=0.5,  # Adjust if needed (0.7-0.9 range)
        iterations=4 # Number of pseudo-labeling iterations
    )
    
    """
    STEP 5: VISUALIZE RESULTS
    """
    print(f"\n📊 Plotting training history...")
    fallacy_hunter.plot_training_history(history)
    
    """
    STEP 6: SAVE THE MODEL
    """
    print(f"\nSaving trained model...")
    fallacy_hunter.save_model('./fallacy_hunter_model')

    print("\n🎯 Finding examples for manual annotation...")
    try:
        # Sample some unused data for uncertainty analysis
        remaining_texts = random.sample(all_unlabeled_texts, min(500, len(all_unlabeled_texts)))
        uncertain_examples = fallacy_hunter.find_uncertain_examples(remaining_texts)
        
        if uncertain_examples:
            # Save to CSV for easy annotation
            uncertain_df = pd.DataFrame(uncertain_examples)
            uncertain_df.to_csv('uncertain_examples_for_annotation.csv', index=False)
            print(f"📝 Saved {len(uncertain_examples)} uncertain examples to 'uncertain_examples_for_annotation.csv'")
            print("These are the best examples to manually annotate for maximum impact!")
    except Exception as e:
        print(f"Could not generate uncertain examples: {e}")

    print("\n🎉 Semi-supervised training completed successfully!")
        
    """
    STEP 7: FINAL EVALUATION (Optional)
    """
    print(f"\n📋 Final training summary:")
    print(f"Initial training examples: {history[0]['train_size']}")
    print(f"Final training examples: {history[-1]['train_size']}")
    print(f"Data expansion: {history[-1]['train_size'] / history[0]['train_size']:.1f}x")
    print(f"Final validation loss: {history[-1]['val_loss']:.4f}")
    
    print("\n🎉 Semi-supervised training completed successfully!")
    print("Your fallacy detection model is ready for browser extension integration!")

if __name__ == "__main__":
    main()