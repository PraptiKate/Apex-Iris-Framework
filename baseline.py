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

# ----------------------------
# 1. Automated Path Logic
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Create the specific 'submission' folder required for the leaderboard
SUBMISSION_DIR = os.path.join(SCRIPT_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Data & Model (Keep your existing logic)
# ----------------------------
iris = load_iris()
X, y = iris.data, (iris.target == 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

class RobustMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.out = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.out(x), dim=1)

model = RobustMLP(input_dim=4, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = F.nll_loss(out, y_train_t)
    loss.backward()
    optimizer.step()

# ----------------------------
# 3. Generating Predictions
# ----------------------------
model.eval()
with torch.no_grad():
    out = model(X_test_t)
    probs = torch.exp(out)[:, 1].cpu().numpy()
preds = (probs >= 0.5).astype(int)

# Create the temporary DataFrame
df_sub = pd.DataFrame({"row_index": range(len(preds)), "target": preds})
temp_csv = os.path.join(SUBMISSION_DIR, "temp.csv")
df_sub.to_csv(temp_csv, index=False)

# ----------------------------
# 4. Encryption (final_submissions.csv.enc)
# ----------------------------
# Generate a key (In a real scenario, you'd store this in GitHub Secrets)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

with open(temp_csv, 'rb') as f:
    raw_data = f.read()

encrypted_data = cipher_suite.encrypt(raw_data)

with open(os.path.join(SUBMISSION_DIR, "final_submissions.csv.enc"), 'wb') as f:
    f.write(encrypted_data)

# Remove the temporary unencrypted CSV for security
os.remove(temp_csv)

# ----------------------------
# 5. Metadata Generation (metadata.json)
# ----------------------------
metadata = {
    "name": "Satyam Anilrao Shelke",
    "PRN": "1132231165",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": "PyTorch RobustMLP",
    "status": "Success"
}

with open(os.path.join(SUBMISSION_DIR, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"DONE: Encrypted submission and metadata generated in {SUBMISSION_DIR}")