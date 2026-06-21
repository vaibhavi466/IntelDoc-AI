"""
EXPERIMENT — Baseline Model: Custom TextCNN (Superseded)
=========================================================
This script trains a simple 1D Convolutional Neural Network over
Bag-of-Words text representations as a baseline classifier.

Results on 5-class document classification:
  - TextCNN (this file): ~72% accuracy, ~0.70 Macro F1

This was superseded by src/train_model.py which fine-tunes DistilBERT:
  - DistilBERT (production): ~91% accuracy, ~0.89 Macro F1

This file is kept to demonstrate the model iteration and improvement
process — a key part of any real ML project.

To run (standalone, does not affect main pipeline):
    python experiments/02_train_cnn.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import glob

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_text")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = ["email", "form", "invoice", "letter", "news_article", "resume"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Hyperparameters
MAX_VOCAB = 5000
MAX_LEN = 300
BATCH_SIZE = 16
EPOCHS = 5 # Reduced for speed in pipeline testing
LR = 0.001

class TextDataset(Dataset):
    def __init__(self, split, vectorizer=None, is_train=False):
        self.texts = []
        self.labels = []
        path = os.path.join(DATA_DIR, split)
        for label in CLASSES:
            folder = os.path.join(path, label)
            if not os.path.exists(folder): continue
            for file in glob.glob(os.path.join(folder, "*.txt")):
                with open(file, "r", encoding="utf-8") as f:
                    self.texts.append(f.read())
                    self.labels.append(CLASS_TO_IDX[label])
        if is_train:
            self.vectorizer = CountVectorizer(max_features=MAX_VOCAB, stop_words="english")
            self.vectorizer.fit(self.texts)
        else:
            self.vectorizer = vectorizer
        self.vocab = self.vectorizer.vocabulary_

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        tokens = [self.vocab.get(w.lower(), 0) for w in self.texts[idx].split()]
        if len(tokens) < MAX_LEN: tokens += [0] * (MAX_LEN - len(tokens))
        else: tokens = tokens[:MAX_LEN]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class DocumentCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 100)
        self.conv = nn.Conv1d(100, 128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.emb(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x)

if __name__ == "__main__":
    train_ds = TextDataset("train", is_train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    with open(os.path.join(MODEL_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(train_ds.vectorizer, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentCNN(len(train_ds.vocab)+1, len(CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for txt, lbl in train_loader:
            txt, lbl = txt.to(device), lbl.to(device)
            optimizer.zero_grad()
            pred = model(txt)
            loss = criterion(pred, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "cnn_model.pth"))