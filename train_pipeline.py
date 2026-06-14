import os, random, numpy as np, librosa, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ── Config ──
# Local training data path — update this to your dataset location
# Training was done on Kaggle using the Fake-or-Real dataset
# kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
DATA_BASE  = "data/"   # place real/ and fake/ folders here to retrain
MODEL_OUT  = "models/deepfake_audio_model.pt"
N_SAMPLES  = 3000      # per class
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

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

def train():
    
    if not os.path.exists(DATA_BASE):
        print("ERROR: Data folder not found.")
        print("Download the dataset from Kaggle and place real/ and fake/ folders inside data/")
        print("Dataset: kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset")
        return
    print(f"Device: {DEVICE}")
    
    REAL_DIR = os.path.join(DATA_BASE, "real")
    FAKE_DIR = os.path.join(DATA_BASE, "fake")
    
    real_files = random.sample(os.listdir(REAL_DIR), N_SAMPLES)
    fake_files = random.sample(os.listdir(FAKE_DIR), N_SAMPLES)
    
    all_paths  = [os.path.join(REAL_DIR, f) for f in real_files] + \
                 [os.path.join(FAKE_DIR, f) for f in fake_files]
    all_labels = [0]*N_SAMPLES + [1]*N_SAMPLES

    print("Extracting features...")
    X, y = [], []
    for path, label in zip(all_paths, all_labels):
        try:
            X.append(extract_melspec(path))
            y.append(label)
        except: pass

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(AudioDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    model = AudioCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((torch.sigmoid(preds)>0.5).float()==yb).sum().item()
            total += len(yb)
        scheduler.step()
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total*100:.2f}%")

    # Evaluate
    model.eval()
    all_preds, all_scores, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            scores = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            all_scores.extend(scores)
            all_preds.extend((scores>0.5).astype(int))
            all_true.extend(yb.numpy().astype(int))

    all_scores = np.array(all_scores)
    all_preds  = np.array(all_preds)
    all_true   = np.array(all_true)
    cm = confusion_matrix(all_true, all_preds)
    fpr, tpr, _ = roc_curve(all_true, all_scores)
    eer = brentq(lambda x: 1.-x-interp1d(fpr,tpr)(x), 0., 1.) * 100

    print("\n" + "="*45)
    print(f"  Accuracy:      {accuracy_score(all_true,all_preds)*100:.2f}%")
    print(f"  F1 Score:      {f1_score(all_true,all_preds)*100:.2f}%")
    print(f"  EER:           {eer:.2f}%")
    print(f"  Real Accuracy: {cm[0,0]/cm[0].sum()*100:.2f}%")
    print(f"  Fake Accuracy: {cm[1,1]/cm[1].sum()*100:.2f}%")
    print("="*45)

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    train()