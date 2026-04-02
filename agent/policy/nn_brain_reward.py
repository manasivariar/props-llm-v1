import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class NNBrainReward:
    def __init__(self, input_dim, lr=1e-3, epochs=100, device="cpu"):
        self.model = MLP(input_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.device = device
        # Initialize as None to detect if training has happened
        self.mu_x, self.std_x = None, None
        self.mu_y, self.std_y = None, None
        self.is_trained = False

    def train_on_batch(self, accumulated_data):
        """Trains on the k*10 candidates collected over 10 iterations."""
        if len(accumulated_data) < 5: return
        
        # Ensure data is numpy for math operations
        X = np.array([np.array(item[0]).flatten() for item in accumulated_data])
        y = np.array([item[1] for item in accumulated_data]).reshape(-1, 1)

        self.mu_x, self.std_x = np.mean(X, axis=0), np.std(X, axis=0) + 1e-6
        self.mu_y, self.std_y = np.mean(y), np.std(y) + 1e-6

        X_t = torch.FloatTensor((X - self.mu_x) / self.std_x).to(self.device)
        y_t = torch.FloatTensor((y - self.mu_y) / self.std_y).to(self.device)

        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_t), y_t)
            loss.backward()
            self.optimizer.step()
        self.is_trained = True

    def predict(self, target_params):
        # 1. Cold start check: If not trained, return a default or random guess
        if not self.is_trained:
            return 0.0 

        # 2. Convert list to numpy array to avoid TypeError 
        target_params = np.array(target_params).flatten()
        
        self.model.eval()
        p_norm = torch.FloatTensor((target_params - self.mu_x) / self.std_x).to(self.device)
        with torch.no_grad():
            pred = self.model(p_norm).item()
        
        return (pred * self.std_y) + self.mu_y