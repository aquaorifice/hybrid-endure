import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cost_model import CostPredictor  # Import the model class


class CostModelTrainer:

    def __init__(self, train_loader, val_loader, test_loader, model, criterion, optimizer, num_epochs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):

            # training
            train_loss = 0.0
            for x_train, y_train in self.train_loader:
                self.optimizer.zero_grad()

                y_pred = self.model(x_train)
                loss = self.criterion(y_pred, y_train)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # validating
            val_loss = 0.0
            for x_val, y_val in self.val_loader:
                y_pred = self.model(x_val)
                loss = self.criterion(y_pred, y_val)
                val_loss += loss.item()


            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train loss: {train_loss/len(self.train_loader):.4f}")
            print(f"Val loss: {val_loss/len(self.val_loader):.4f}")

    def evaluate(self, data_loader):
        # Evaluation loop
        test_loss = 0.0
        for x_test, y_test in data_loader:
            y_pred = self.model(x_test)
            loss = self.criterion(y_pred, y_test)
            test_loss += loss.item()

        return test_loss / len(data_loader)

