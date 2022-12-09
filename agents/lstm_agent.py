import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import copy
from typing import Union

from agents.base_agent import BaseAgent

class LSTMAgent(BaseAgent):
    def __init__(
        self,
        baseline_reward: float,
        max_seq_len: int = 128,
        embedding_dim: int = 32,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        initial_lr: float = 1e-3
    ):
        super().__init__()
        self.baseline_reward = baseline_reward
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.initial_lr = initial_lr

        # Input size: (seq_len, batch_size)
        # Output size: (seq_len, batch_size, 13)
        self.embedding = nn.Embedding(13, embedding_dim).cuda()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            proj_size=13
        ).cuda()

        self.optimizer = optim.Adam(
            list(self.embedding.parameters()) + list(self.lstm.parameters()),
            lr=initial_lr
        )

    def log_prob(self, heads: torch.LongTensor) -> torch.Tensor:
        # Find lengths of each head and create a mask
        assert heads.size(0) % 4 == 0
        layer_types = heads[torch.arange(0, heads.size(0), 4)]
        layer_num_params = self._num_params_per_layer[layer_types]
        parameter_mask = self._parameter_mask[layer_num_params.T] \
                             .reshape((heads.size(1), -1)).T

        # Note: Using the fact that LayerType.End is largest
        end_indices = layer_types.argmax(dim=0) * 4
        mask = torch.zeros_like(heads)
        mask[end_indices, torch.arange(heads.size(1))] = 1.
        mask = (1 - mask.cumsum(dim=0))[:-1] * parameter_mask[1:]

        embeddings = self.embedding(heads)
        logits, _ = self.lstm(embeddings)

        # Compute total log probability of these heads
        log_probs = logits - torch.logsumexp(logits, dim=-1).unsqueeze(-1)
        log_probs_per_layer = torch.gather(
            log_probs[:-1],
            dim=-1,
            index=heads[1:].unsqueeze(-1)
        ).squeeze(-1) * mask
        log_probs_per_head = log_probs_per_layer.sum(dim=0)

        return log_probs_per_head

    def sample_heads(self, batch_size: int) -> torch.LongTensor:
        curr_layer = torch.zeros((4, batch_size)).long().cuda()
        seq_len = 4
        layers = [curr_layer]
        embeddings = self.embedding(curr_layer)
        logits, (h, c) = self.lstm(embeddings)
        logits = logits[-1:]

        has_ended = torch.zeros((batch_size, ), dtype=torch.bool).cuda()
        patch_zeros = 4

        with torch.no_grad():
            while seq_len < self.max_seq_len - 4:
                distribution = Categorical(logits=logits)
                curr_layer = distribution.sample() * (~has_ended).long()
                layers.append(curr_layer)

                if seq_len % 4 == 0:
                    has_ended = torch.logical_or(has_ended, curr_layer == 12)
                    if torch.all(has_ended):
                        patch_zeros = 3
                        break

                embeddings = self.embedding(curr_layer)
                logits, (h, c) = self.lstm(embeddings, (h, c))

                seq_len += 1

        curr_layer = torch.zeros((patch_zeros, batch_size)).long().cuda()
        curr_layer[0] = 12
        layers.append(curr_layer)
        heads = torch.cat(layers, dim=0)

        # Apply parameter mask to sampled heads
        layer_types = heads[torch.arange(0, heads.size(0), 4)]
        layer_num_params = self._num_params_per_layer[layer_types]
        parameter_mask = self._parameter_mask[layer_num_params.T].long() \
                             .reshape((heads.size(1), -1)).T
        heads *= parameter_mask
        return heads

    def update(
        self,
        heads: torch.LongTensor,
        rewards: torch.Tensor,
        on_policy: bool,
        heads_log_prob: Union[torch.Tensor, None] = None
    ) -> dict:
        current_log_prob = self.log_prob(heads)
        if on_policy:
            J = current_log_prob * (rewards - self.baseline_reward)
        else:
            J = torch.exp(current_log_prob - heads_log_prob) \
                * (rewards - self.baseline_reward)
        loss = -J.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            "LSTM Agent Loss": loss.item(),
            "Average Reward": rewards.mean().item()
        }
        return loss_dict

    def save(self, path: str):
        embedding = copy.deepcopy(self.embedding).cpu()
        lstm = copy.deepcopy(self.lstm).cpu()
        model_dict = {
            "baseline_reward": self.baseline_reward,
            "max_seq_len": self.max_seq_len,
            "embedding_dim": self.embedding_dim,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_num_layers": self.lstm_num_layers,
            "initial_lr": self.initial_lr,
            "embedding": embedding.state_dict(),
            "lstm": lstm.state_dict()
        }
        torch.save(model_dict, path)

    # TODO: change this to a static method that returns an instance
    def load(self, path: str):
        model_dict = torch.load(path, map_location="cuda")

        self.baseline_reward = model_dict["baseline_reward"]
        self.max_seq_len = model_dict["max_seq_len"]
        self.embedding.load_state_dict(model_dict["embedding"])
        self.lstm.load_state_dict(model_dict["lstm"])

        self.embedding.cuda()
        self.lstm.cuda()

    # See train/train_baseline.py:LayerType enumeration for more details
    _num_params_per_layer = torch.tensor(
        [0, 2, 3, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0]
    ).cuda()

    _parameter_mask = torch.tensor([
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]
    ]).cuda()
