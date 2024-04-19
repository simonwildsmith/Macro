import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


"""
Step 1: Load the data
"""

# Load the data
data_path = "datasets/cleaned/merged.csv"
data = pd.read_csv(data_path)
print(data.head())

"""
Step 2: Preprocess the data
"""

# Drop missing values
if data.isnull().values.any():
    num_rows_before = data.shape[0]
    data = data.dropna()
    num_rows_after = data.shape[0]
    num_rows_removed = num_rows_before - num_rows_after
    print(f"Number of rows removed: {num_rows_removed}")


# Split the data into features and target
X = data.drop(columns=["Date", "Historical Gold Prices_cleaned"])
y = data["Historical Gold Prices_cleaned"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # Convert data into PyTorch tensors
    train_features = torch.tensor(X_train, dtype=torch.float32)
    train_targets = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    val_features = torch.tensor(X_val, dtype=torch.float32)
    val_targets = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # DataLoaders
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Model initialization
    input_size = X_train.shape[1]
    model = LinearRegressionModel(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(50):  # number of epochs can be adjusted
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Validation loss
    val_loss = evaluate_model(model, val_loader, criterion)
    return val_loss


def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


original_features = X.columns.tolist()
loss_results = {}
