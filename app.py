import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import tempfile
import os
import json

# ── Model definition (must match exactly what we trained) ──
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ── Load model once ──
@st.cache_resource
def load_model():
    model = AudioCNN()
    model.load_state_dict(torch.load(
        'models/deepfake_audio_model.pt',
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

# ── Feature extraction (same as training) ──
def extract_melspec(path, sr=16000, n_mels=64, duration=2.0):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)

# ── Predict ──
def predict(path, model):
    feat = extract_melspec(path)
    inp = torch.tensor(feat).unsqueeze(0).unsqueeze(0)  # (1,1,64,63)
    with torch.no_grad():
        score = torch.sigmoid(model(inp)).item()
    label = "Deepfake (AI-Generated)" if score > 0.5 else "Genuine (Human)"
    confidence = score if score > 0.5 else 1 - score
    return label, confidence, score

# ── UI ──
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Deepfake Audio Detector")
st.markdown("Upload a speech audio file to detect whether it is **Genuine (Human)** or **Deepfake (AI-Generated)**.")
st.divider()

model = load_model()

uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "flac", "mp3"],
    help="Supported formats: WAV, FLAC, MP3"
)

if uploaded:
    # Save to temp file
    suffix = "." + uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    st.audio(tmp_path)
    st.divider()

    with st.spinner("Analyzing audio..."):
        try:
            label, confidence, raw_score = predict(tmp_path, model)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            os.unlink(tmp_path)
            st.stop()

    # Result
    if "Deepfake" in label:
        st.error(f"## 🔴 {label}")
    else:
        st.success(f"## 🟢 {label}")

    # Confidence bar
    st.metric("Confidence", f"{confidence*100:.1f}%")
    st.progress(confidence)

    # Details
    with st.expander("See details"):
        st.write(f"**Raw score:** {raw_score:.4f} (>0.5 = Deepfake)")
        st.write(f"**File:** {uploaded.name}")
        st.write(f"**Model:** CNN on Mel-spectrogram (64 mels, 2s window)")

    os.unlink(tmp_path)

st.divider()
st.caption("MARS Open Projects 2026 | Deepfake Audio Detection | Model accuracy: 99.58%")