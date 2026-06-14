import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
import json
import csv
import os

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

def load_model(model_path='models/deepfake_audio_model.pt'):
    model = AudioCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

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

def predict_single(path, model):
    feat = extract_melspec(path)
    inp = torch.tensor(feat).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = torch.sigmoid(model(inp)).item()
    label = "Deepfake (AI-Generated)" if score > 0.5 else "Genuine (Human)"
    confidence = score if score > 0.5 else 1 - score
    return label, confidence, score

def predict_csv(csv_path, model):
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row['filepath']
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            label, confidence, score = predict_single(path, model)
            results.append({
                'filepath': path,
                'prediction': label,
                'confidence': f"{confidence*100:.2f}%",
                'raw_score': f"{score:.4f}"
            })
            print(f"{os.path.basename(path):40s} → {label} ({confidence*100:.1f}%)")
    
    out_path = csv_path.replace('.csv', '_predictions.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepfake Audio Detector')
    parser.add_argument('--input', required=True, help='Path to .wav file or .csv with filepaths')
    parser.add_argument('--model', default='models/deepfake_audio_model.pt')
    args = parser.parse_args()

    model = load_model(args.model)
    print(f"Model loaded from {args.model}")

    if args.input.endswith('.csv'):
        predict_csv(args.input, model)
    else:
        label, confidence, score = predict_single(args.input, model)
        print(f"\nFile: {args.input}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Raw score:  {score:.4f}")