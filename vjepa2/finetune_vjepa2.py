import sys
sys.path.append('/gpfs/home/preiyalt/vjepa2')
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

from src.models.vision_transformer import vit_large

CHECKPOINT = '/gpfs/projects/infoseeking/preiyalt/checkpoints/vitl.pt'
VIDEOS_BASE = '/gpfs/projects/infoseeking/preiyalt/Main/'
METADATA = '/gpfs/projects/infoseeking/preiyalt/Main/metadata.csv'
OUT_DIR = '/gpfs/projects/infoseeking/preiyalt/rq3_outputs/vjepa2/'
SEQ_LEN = 16
RESOLUTION = 224
EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading V-JEPA 2 model...")
model = vit_large(img_size=224, patch_size=16, tubelet_size=2, uniform_power=True, use_rope=True)
checkpoint = torch.load(CHECKPOINT, map_location=device)
state_dict = checkpoint.get("target_encoder", checkpoint)
model.load_state_dict(state_dict, strict=False)

for name, param in model.named_parameters():
    if not any(x in name for x in ["blocks.22", "blocks.23", "norm"]):
        param.requires_grad = False
    else:
        param.requires_grad = True

class FineTunedVJEPA(nn.Module):
    def __init__(self, backbone, num_classes=4):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.mean(dim=1)
        return self.classifier(feat)

ft_model = FineTunedVJEPA(model, num_classes=4).to(device)
trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in ft_model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,}")

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, SEQ_LEN, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    if len(frames) < SEQ_LEN:
        return None
    return frames

df = pd.read_csv(METADATA)
df_id = df[df["type"] == "1_Possible"].reset_index(drop=True)

print("Loading frames...")
X_frames = []
y_labels = []
for i, row in df_id.iterrows():
    vid_path = os.path.join(VIDEOS_BASE, row["file_name"])
    if not os.path.exists(vid_path):
        continue
    frames = load_frames(vid_path)
    if frames is None:
        continue
    X_frames.append(frames)
    y_labels.append(row["condition"])

le = LabelEncoder()
y_enc = le.fit_transform(y_labels)
print(f"Loaded {len(X_frames)} scenes, classes: {list(le.classes_)}")

def frames_to_tensor(frames):
    tensor = torch.tensor(np.stack(frames).astype(np.float32) / 255.0)
    return tensor.permute(0, 3, 1, 2)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X_frames, y_enc)):
    print(f"Fold {fold+1}/5")
    ft_model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ft_model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        np.random.shuffle(train_idx)
        epoch_loss = 0
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch_idx = train_idx[i:i+BATCH_SIZE]
            batch_labels = torch.tensor(y_enc[batch_idx], dtype=torch.long).to(device)
            all_feats = []
            for j in batch_idx:
                frames_tensor = frames_to_tensor(X_frames[j]).to(device)
                logits = ft_model(frames_tensor)
                all_feats.append(logits.mean(0))
            batch_logits = torch.stack(all_feats)
            loss = criterion(batch_logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, loss: {epoch_loss:.4f}")

    ft_model.eval()
    preds = []
    with torch.no_grad():
        for j in test_idx:
            frames_tensor = frames_to_tensor(X_frames[j]).to(device)
            logits = ft_model(frames_tensor)
            pred = logits.mean(0).argmax().item()
            preds.append(pred)
    acc = accuracy_score(y_enc[test_idx], preds)
    fold_accs.append(acc)
    print(f"  Fold {fold+1} accuracy: {acc:.3f}")

fold_accs = np.array(fold_accs)
mean_acc = fold_accs.mean()
std_acc = fold_accs.std()
t_stat, p_val = stats.ttest_1samp(fold_accs, 0.25)

print(f"Condition accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
print(f"t={t_stat:.3f}, p={p_val:.4f}")

results = {
    "model": "V-JEPA 2",
    "task": "condition",
    "mean": round(float(mean_acc), 4),
    "std": round(float(std_acc), 4),
    "fold_accs": fold_accs.tolist(),
    "t_stat": round(float(t_stat), 4),
    "p_value": round(float(p_val), 4),
    "significant": bool(p_val < 0.05),
    "chance": 0.25
}

with open(OUT_DIR + "finetune_results_vjepa2.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved.")
