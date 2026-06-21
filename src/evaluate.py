import json
import os
from seaborn import cm
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MODEL_DIR, DATA_CSV, CONFUSION_MATRIX, EVAL_METRICS

# CONFIG
MODEL_PATH = str(MODEL_DIR)
DATA_PATH  = str(DATA_CSV)

def evaluate():
    print(" Loading Model & Test Data...")
    
    # 1. Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval() # Set to evaluation mode
    except (OSError, ValueError) as e:
        print(f"Model loading failed: {e}")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=['text'])
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Create Labels
    label_list = df['category'].unique().tolist()
    label_list.sort()
    label2id = {label: i for i, label in enumerate(label_list)}
    
    # Filter for TEST split only (Data the model hasn't seen)
    test_df = df[df['split'] == 'test']
    
    if len(test_df) == 0:
        print("Warning: No test documents found in evaluation dataset.")
        return
        
    # Optimization: Use a smaller sample for speed if dataset is huge
    if len(test_df) > 500:
        test_df = test_df.sample(500, random_state=42)
        
    print(f"Evaluating on {len(test_df)} test documents...")

    # 3. Run Predictions
    predictions = []
    true_labels = []
    
    print(" Running Inference...")
    for index, row in test_df.iterrows():
        text = row['text']
        true_label_id = label2id[row['category']]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            
        predictions.append(pred_id)
        true_labels.append(true_label_id)

    # 4. Calculate Metrics
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=label_list, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']

    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT")
    print("="*30)
    print(f" Accuracy:  {acc:.2%}")
    print(f" Macro F1:  {macro_f1:.2f}")
    print("-" * 30)
    print(classification_report(true_labels, predictions, target_names=label_list))

    # 5. Save eval metrics to JSON 
    cm = confusion_matrix(true_labels, predictions) 
    os.makedirs(os.path.dirname(EVAL_METRICS), exist_ok=True)

    with open(EVAL_METRICS, "w") as f:
        json.dump({
            "accuracy":         round(acc, 4),
            "macro_f1":         round(macro_f1, 4),
            "confusion_matrix": cm.tolist(),          
            "class_report":     report,               
        }, f, indent=2)

    print(f"Metrics saved to: {EVAL_METRICS}")
      

    plt.figure(figsize=(10, 8))                       
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_list,
        yticklabels=label_list
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix — IntelDoc AI')
    plt.tight_layout()

    os.makedirs(os.path.dirname(str(CONFUSION_MATRIX)), exist_ok=True)
    plt.savefig(str(CONFUSION_MATRIX), dpi=150, bbox_inches='tight')  
    plt.close()
    print(f"Confusion Matrix saved to: {CONFUSION_MATRIX}")

if __name__ == "__main__":
    evaluate()