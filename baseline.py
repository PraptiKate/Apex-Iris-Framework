import os
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cryptography.fernet import Fernet
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------------
# 1. Dynamic Automated Path Logic & Ghost Filtering
# ----------------------------
BLOCK_LIST = ["Satyam_Anilrao_Shelke", "SatyamShelke2005", "Test_User"]

submitter_raw = os.getenv('SUBMITTER_NAME', 'Satyam_Anilrao_Shelke')

if any(blocked_name in submitter_raw for blocked_name in BLOCK_LIST):
    print(f"!!! CRITICAL: Blocking execution for {submitter_raw} !!!")
    exit(0)

clean_name = submitter_raw.replace(" ", "_").replace(".", "_")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(SCRIPT_DIR, "submissions", clean_name)
DATA_JSON_PATH = os.path.join(SCRIPT_DIR, "docs", "data.json")

os.makedirs(SUBMISSION_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Data Preparation
# ----------------------------
iris = load_iris()
X, y = iris.data, (iris.target == 1).astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)

X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.LongTensor(y_val).to(device)

X_test_t = torch.FloatTensor(X_test).to(device)

# ----------------------------
# 3. Improved Model
# ----------------------------
class RobustMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = RobustMLP(input_dim=4, num_classes=2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# ----------------------------
# 4. Training with Early Stopping
# ----------------------------
best_val_loss = float('inf')
patience = 10
counter = 0

print(f"--- Training Improved Model for {submitter_raw} ---")

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered!")
        break

# ----------------------------
# 5. Predictions & Metrics
# ----------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

preds = (probs >= 0.5).astype(int)

accuracy_val = np.mean(preds == y_test) * 100
f1_val = f1_score(y_test, preds, average='weighted') * 100
precision_val = precision_score(y_test, preds) * 100
recall_val = recall_score(y_test, preds) * 100

print(f"\nAccuracy: {accuracy_val:.2f}%")
print(f"F1 Score: {f1_val:.2f}%")
print(f"Precision: {precision_val:.2f}%")
print(f"Recall: {recall_val:.2f}%")

# ----------------------------
# 6. Save Predictions
# ----------------------------
df_sub = pd.DataFrame({"row_index": range(len(preds)), "target": preds})
temp_csv = os.path.join(SUBMISSION_DIR, "temp.csv")
df_sub.to_csv(temp_csv, index=False)

# ----------------------------
# 7. Encryption
# ----------------------------
key = Fernet.generate_key()
cipher_suite = Fernet(key)

with open(temp_csv, 'rb') as f:
    raw_data = f.read()

encrypted_data = cipher_suite.encrypt(raw_data)

with open(os.path.join(SUBMISSION_DIR, "final_submissions.csv.enc"), 'wb') as f:
    f.write(encrypted_data)

os.remove(temp_csv)

# ----------------------------
# 8. Metadata
# ----------------------------
display_name = submitter_raw

if "Satyam" in submitter_raw or "Shelke" in submitter_raw:
    prn = "1132231165"
else:
    prn = "EXTERNAL_CONTRIBUTOR"

metadata = {
    "name": display_name,
    "PRN": prn,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "Improved PyTorch RobustMLP",
    "status": "Success",
    "accuracy": f"{accuracy_val:.2f}%",
    "precision": f"{precision_val:.2f}%",
    "recall": f"{recall_val:.2f}%",
    "f1_score": f"{f1_val:.2f}%",
    "submission_type": "Automated_CI_CD"
}

with open(os.path.join(SUBMISSION_DIR, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

# ----------------------------
# 9. Leaderboard Update
# ----------------------------
new_entry = {
    "Participant": display_name,
    "Architecture": "Improved PyTorch RobustMLP",
    "Accuracy": f"{accuracy_val:.1f}%",
    "F1-Score": f"{f1_val:.1f}",
    "Timestamp": datetime.now().strftime("%Y-%m-%d")
}

try:
    if os.path.exists(DATA_JSON_PATH):
        with open(DATA_JSON_PATH, 'r') as f:
            leaderboard_data = json.load(f)
    else:
        leaderboard_data = []

    NAMES_TO_SCRUB = ["Satyam Anilrao Shelke", "Satyam Shelke", "Test_User"]

    leaderboard_data = [
        e for e in leaderboard_data
        if e.get("Participant") not in NAMES_TO_SCRUB
        and e.get("Participant") != display_name
    ]

    leaderboard_data.append(new_entry)

    with open(DATA_JSON_PATH, 'w') as f:
        json.dump(leaderboard_data, f, indent=4)

    print(f"\nLeaderboard updated for: {display_name}")

except Exception as e:
    print(f"\nLeaderboard update failed: {e}")

print("\n--- PROCESS COMPLETE ---")
