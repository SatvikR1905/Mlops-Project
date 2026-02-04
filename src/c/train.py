import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from model import CancerNet

VERSION = "v1"
os.makedirs(f"models/{VERSION}", exist_ok=True)

df = pd.read_csv("data/processed/train.csv")

X = torch.tensor(df.drop("target", axis=1).values, dtype=torch.float32)
y = torch.tensor(df["target"].values, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CancerNet(X.shape[1])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb).squeeze(), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), f"models/{VERSION}/model.pt")
print("Model trained & saved")
