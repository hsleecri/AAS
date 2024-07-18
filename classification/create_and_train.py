import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx

# 데이터셋 생성
np.random.seed(0)
X = np.random.rand(100, 3).astype(np.float32)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(np.float32)

# PyTorch 데이터셋 및 데이터로더
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = Dataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = MLP()

# 모델 학습
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

# 모델 저장 (ONNX 형식)
dummy_input = torch.randn(1, 3)
torch.onnx.export(model, dummy_input, "mlp_classification.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("MLP 분류 모델이 'mlp_classification.onnx'에 저장되었습니다.")
