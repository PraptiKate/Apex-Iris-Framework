import os
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Paths
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "submissions")
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Iris dataset (The Teacher's Request)
# ----------------------------
iris = load_iris()
X = iris.data
y = (iris.target == 1).astype(int) # Binary: 1 vs 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

# ----------------------------
# MLP Model (Tabular Data)
# ----------------------------
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

# ----------------------------
# Training (Note the New Label)
# ----------------------------
print("--- STARTING IRIS DATA TRAINING ---")

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = F.nll_loss(out, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Iris Epoch {epoch+1} | Loss {loss.item():.4f}")

# ----------------------------
# Diverse Prediction Logic
# ----------------------------
def get_diverse_preds(model, data_t, target_percent=40):
    model.eval()
    with torch.no_grad():
        out = model(data_t)
        probs = torch.exp(out)[:, 1].cpu().numpy()
    
    threshold = np.percentile(probs, 100 - target_percent)
    return (probs >= threshold).astype(int).tolist()

print("Generating IRIS predictions...")
ideal_preds = get_diverse_preds(model, X_test_t, target_percent=35)

# ----------------------------
# Save Submissions
# ----------------------------
pd.DataFrame({"row_index": range(len(ideal_preds)), "target": ideal_preds}).to_csv(
    os.path.join(SUBMISSIONS_DIR, "ideal_submission.csv"), index=False)
pd.DataFrame({"row_index": range(len(ideal_preds)), "target": ideal_preds}).to_csv(
    os.path.join(SUBMISSIONS_DIR, "perturbed_submission.csv"), index=False)

print("-" * 30)
print("SUCCESS: Iris files generated with 1s and 0s.")
print("-" * 30)