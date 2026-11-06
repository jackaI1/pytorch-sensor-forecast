"""Direct multi-horizon tree and sequence models with explicit CPU seeding."""

import random

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_boosting(x: np.ndarray, y: np.ndarray, seed: int) -> MultiOutputRegressor:
    # No randomized validation split or early stopping on future observations.
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    )
    return model.fit(x, y)


class SmallLSTM(nn.Module):
    def __init__(self, outputs: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])


class LSTMForecaster:
    def __init__(self, seed: int = 42, epochs: int = 12):
        if epochs < 1:
            raise ValueError("epochs must be positive")
        self.seed = seed
        self.epochs = epochs

    def fit(self, x: np.ndarray, y: np.ndarray, observed: np.ndarray):
        seed_everything(self.seed)
        # Caller passes only the current fold's pre-cutoff training observations.
        self.scaler = StandardScaler().fit(observed.reshape(-1, 1))
        tx = torch.tensor(
            self.scaler.transform(x.reshape(-1, 1)).reshape(*x.shape, 1), dtype=torch.float32
        )
        ty = torch.tensor(
            self.scaler.transform(y.reshape(-1, 1)).reshape(y.shape), dtype=torch.float32
        )
        self.model = SmallLSTM(y.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        self.losses = []
        self.model.train()
        for _ in range(self.epochs):
            total = 0.0
            for batch in torch.randperm(len(tx)).split(128):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model(tx[batch]), ty[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(batch)
            self.losses.append(total / len(tx))
        self.model.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        tx = torch.tensor(
            self.scaler.transform(x.reshape(-1, 1)).reshape(*x.shape, 1), dtype=torch.float32
        )
        with torch.inference_mode():
            predictions = self.model(tx).numpy()
        return self.scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
