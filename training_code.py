# ============================================================
# FAST FLUX DETECTION - TRAINING PIPELINE (LSTM)
# ============================================================

# ----------------------------
# IMPORTS
# ----------------------------
import os
import re
import math
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from collections import Counter
from statistics import mean, stdev

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# CONFIG
# ----------------------------
SEQ_LEN = 5
BATCH_SIZE = 32
EPOCHS = 30
HIDDEN_SIZE = 64
LR = 0.001
DEVICE = "cpu"   # FORCE CPU FOR CONSISTENCY

# CHANGE THESE TO YOUR DATASET PATHS
LEGIT_DIR = r"C:\Users\09shi\Desktop\UIA LEARNING MATERIAL\FYP\newdataset\archive2\Fastflux Attack Dataset\benign"
FASTFLUX_DIR = r"C:\Users\09shi\Desktop\UIA LEARNING MATERIAL\FYP\newdataset\archive2\Fastflux Attack Dataset\ff"

OUTPUT_DIR = "fastflux-demo"   # where app.py lives
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def shannon_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0:
        return 0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def extract_features(dig_text):
    a_records = re.findall(r'IN\s+A\s+(\d+\.\d+\.\d+\.\d+)', dig_text)
    ttl_vals = list(map(int, re.findall(r'(\d+)\s+IN\s+A\s+', dig_text)))
    cname_records = re.findall(r'IN\s+CNAME\s+(\S+)', dig_text)
    ns_records = re.findall(r'IN\s+NS\s+(\S+)', dig_text)

    subnets = {'.'.join(ip.split('.')[:3]) for ip in a_records}

    return [
        len(a_records),
        min(ttl_vals) if ttl_vals else 0,
        max(ttl_vals) if ttl_vals else 0,
        mean(ttl_vals) if ttl_vals else 0,
        stdev(ttl_vals) if len(ttl_vals) > 1 else 0,
        len(cname_records),
        len(ns_records),
        shannon_entropy(a_records),
        len(subnets)
    ]

# ============================================================
# LOAD DATASET
# ============================================================
records = []

def load_folder(folder, label):
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), encoding="utf-8", errors="ignore") as f:
                content = f.read()
                entries = content.split("; <<>> DiG")
                for entry in entries:
                    if entry.strip():
                        records.append({
                            "features": extract_features(entry),
                            "label": label
                        })

print("Loading dataset...")
load_folder(LEGIT_DIR, 0)
load_folder(FASTFLUX_DIR, 1)

df = pd.DataFrame(records)
X = np.array(df["features"].to_list())
y = np.array(df["label"].to_list())

print("Total samples:", len(y))
print("Class distribution:", np.bincount(y))

# ============================================================
# SCALING
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SAVE SCALER FOR STREAMLIT APP
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
print("Scaler saved ✔")

# ============================================================
# SEQUENCE CREATION
# ============================================================
def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

X_seq, y_seq = make_sequences(X_scaled, y, SEQ_LEN)

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.3, stratify=y_seq, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                  torch.tensor(y_val, dtype=torch.long)),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.long)),
    batch_size=BATCH_SIZE
)

# ============================================================
# MODEL
# ============================================================
class LSTM(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.fc = nn.Linear(h, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = LSTM(d=9, h=HIDDEN_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ============================================================
# TRAINING LOOP
# ============================================================
print("Training LSTM...")
best_val = float("inf")
patience = 5
counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += loss_fn(model(xb), yb).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train={train_loss:.4f}  Val={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(
            model.state_dict(),
            os.path.join(OUTPUT_DIR, "model_lstm.pth")
        )
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

print("Model saved ✔")

# ============================================================
# FINAL EVALUATION
# ============================================================
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "model_lstm.pth")))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        y_true.extend(yb.numpy())
        y_pred.extend(preds.numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nTraining complete ✅")
