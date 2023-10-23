import os
import pandas as pd
import toml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cost_model import CostPredictor
from train_cost_model import CostModelTrainer


class CostModel:
    def __init__(self):
        file_dir = os.path.dirname(__file__)
        config_path = os.path.join(file_dir, "config.toml")
        with open(config_path) as fid:
            self.config = toml.load(fid)
        self.data_dir = self.config["job"]["LCMDataGen"]["train_data_dir"]

    def read_train_file(self, dir):
        train_data = []
        for filename in os.listdir(dir):
            if filename.endswith('.parquet'):
                parquet_file = os.path.join(dir, filename)
                df = pd.read_parquet(parquet_file)
                train_data.append(df)

        combined_train_data = pd.concat(train_data, ignore_index=True)
        return combined_train_data


if __name__ == "__main__":
    batch_size = 64
    cost_model = CostModel()
    train_data = cost_model.read_train_file(cost_model.data_dir)
    data = train_data[["z0", "z1", "q", "w", "B", "s", "E"]].values
    target = train_data[["z0_cost", "z1_cost", "q_cost", "w_cost"]].sum(axis=1).values
    data = torch.tensor(data, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    train_data, val_data, test_data = torch.utils.data.random_split(data, [0.6, 0.2, 0.2])
    train_loader = DataLoader(TensorDataset(data, target), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(data, target), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(data, target), batch_size=batch_size)


# Model
    model = CostPredictor(data.shape[1])

# Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = CostModelTrainer(train_loader, val_loader, test_loader, model, criterion, optimizer, num_epochs=100)
    trainer.train()
    test_loss = trainer.evaluate(test_loader)
    print(f"Test loss: {test_loss:.4f}")
    trainer.save_model('cost_predictor_model.pth')








