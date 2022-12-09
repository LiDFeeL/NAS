import torch
import torch.nn as nn
import torch.optim as optim

from exploration.base_exploration_model import BaseExplorationModel

class RNDModel(BaseExplorationModel):
    def __init__(
        self,
        embedding_dim: int = 32,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        output_dim: int = 16,
        initial_lr: float = 1e-3
    ):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Embedding(13, embedding_dim),
            nn.LSTM(
                input_size=embedding_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                proj_size=output_dim
            )
        ).cuda()
        self.net2 = nn.Sequential(
            nn.Embedding(13, embedding_dim),
            nn.LSTM(
                input_size=embedding_dim,
                hidden_size=lstm_hidden_dim,
                proj_size=output_dim
            )
        ).cuda()

        # Initialize parameters for the two networks different so RND works
        # better
        for weight in self.net1.parameters():
            nn.init.normal_(weight, 0, output_dim**-0.5)
        for weight in self.net2.parameters():
            nn.init.uniform_(weight, -output_dim**-0.5, output_dim**-0.5)
        
        self.optimizer = optim.Adam(self.net2.parameters(), lr=initial_lr)

    def forward(self, heads: torch.LongTensor):
        out1, _ = self.net1(heads)
        out1.detach()
        out2, _ = self.net2(heads)
        error = torch.mean((out1 - out2) ** 2, dim=[0, 2]) ** 0.5
        return error

    def update(self, heads: torch.LongTensor):
        error = self.forward(heads)
        loss = error.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            "Exploration Reward": error.detach(),
            "Exploration Loss": loss.item()
        }
        return loss_dict
