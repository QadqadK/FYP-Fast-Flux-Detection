# ============================================================
# FAST FLUX DETECTION - STREAMLIT INTERFACE
# ============================================================

import streamlit as st
import torch
import os
import joblib
import numpy as np

from feature_extractor import extract_features
from model import LSTM

# ----------------------------------
# CONFIG
# ----------------------------------
SEQ_LEN = 5
DEVICE = "cpu"  # Force CPU for Streamlit

# ----------------------------------
# PATHS (relative to app.py)
# ----------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_lstm.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Safety check
assert os.path.exists(MODEL_PATH), f"❌ Missing {MODEL_PATH}"
assert os.path.exists(SCALER_PATH), f"❌ Missing {SCALER_PATH}"

# ----------------------------------
# LOAD MODEL & SCALER
# ----------------------------------
model = LSTM(d=9, h=64)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

scaler = joblib.load(SCALER_PATH)

# ----------------------------------
# STREAMLIT UI
# ----------------------------------
st.title("🔐 Fast Flux Detection Simulator")
st.write("DNS-based Fast Flux Attack Detection using RNN (LSTM)")

uploaded_file = st.file_uploader("Upload DNS Log (.txt)", type=["txt"])

if uploaded_file:
    dns_text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Split into dig responses
    entries = dns_text.split("; <<>> DiG")

    # Extract features
    feature_seq = []
    for entry in entries:
        if entry.strip():
            feature_seq.append(extract_features(entry))

    if len(feature_seq) < SEQ_LEN:
        st.error(f"❌ Not enough DNS responses (need at least {SEQ_LEN} dig blocks).")
    else:
        # Take only first SEQ_LEN responses
        feature_seq = feature_seq[:SEQ_LEN]

        # Scale features
        X_scaled = scaler.transform(feature_seq)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # shape: [1, seq_len, features]

        if st.button("🚀 Detect Fast Flux"):
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            # Determine label
            label = "🚨 Fast Flux Attack" if probs[1] > probs[0] else "✅ Benign Domain"

            # Show results
            st.subheader("🔎 Detection Result")
            st.success(label)

            st.subheader("📊 Confidence")
            st.bar_chart({
                "Benign": float(probs[0]),
                "Fast Flux": float(probs[1])
            })

            st.subheader("🧠 Extracted Feature Sequence")
            st.dataframe(feature_seq)
