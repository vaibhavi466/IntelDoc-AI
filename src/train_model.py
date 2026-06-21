"""
train_model.py — Fine-tunes DistilBERT for document classification.

Pipeline:
  1. Load CSV produced by create_dataset.py
  2. Tokenize text with DistilBERT tokenizer
  3. Fine-tune with HuggingFace Trainer API
  4. Log all params + metrics to MLflow
  5. Save model weights to MODEL_DIR
"""

import os
import mlflow
import pandas as pd
import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, DatasetDict

from src.config import BASE_MODEL_NAME, DATA_CSV, MODEL_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MAX_SEQ_LENGTH


mlflow.set_experiment("IntelDoc_Experiments")

# 1. SETUP CONFIGURATION
# We switch to DistilBERT, which is lighter and faster for CPU training
MODEL_NAME = BASE_MODEL_NAME
DATA_PATH  = str(DATA_CSV)
OUTPUT_DIR = str(MODEL_DIR)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    _, _, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }

def main():
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    # 1. Load Data
    print("Loading Dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Remove empty rows
    df = df.dropna(subset=['text'])
    
    # Create Labels
    label_list = df['category'].unique().tolist()
    label_list.sort() # Ensure consistent order
    num_labels = len(label_list)
    
    # Create id2label mappings
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    print(f"Categories found: {label_list}")
    
    # Map text categories to numbers in the dataframe
    df['label'] = df['category'].map(label2id)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    ####### Dataset Split: 80% Train, 20% Test
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].reset_index(drop=True)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test":  Dataset.from_pandas(val_df)
    })
    
    # 2. Tokenization
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # The tokenizer converts text into tokens (numbers)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True) #Tokenize the full dataset

    # 3. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Training Arguments : hyperparameters(training settings) are defined using TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,  #standard for transformers
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        report_to="mlflow",
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 6. Train
    with mlflow.start_run():                           # ← FIX: actual MLflow run
        # Log all hyperparameters
        mlflow.log_params({
            "model_name":      BASE_MODEL_NAME,
            "max_seq_length":  MAX_SEQ_LENGTH,
            "batch_size":      BATCH_SIZE,
            "learning_rate":   LEARNING_RATE,
            "num_epochs":      NUM_EPOCHS,
            "num_classes":     num_labels,
            "classes":         str(label_list),
        })

        print("Starting Training...")
        trainer.train()

        # Log evaluation metrics to MLflow
        eval_results = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_results.items() if isinstance(v, float)})
        
        # 7. Save
        print("Saving Final Model...")
        trainer.save_model(OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()