import torch

from typing import Union

from train.train_custom_head import Head

class BaseAgent:
    def __init__(self):
        super().__init__()
    
    def log_prob(self, heads: torch.LongTensor) -> torch.Tensor:
        """
        Find the log probabilities of some proposed heads.
        """
        raise NotImplementedError()

    def sample_heads(self, batch_size: int) -> torch.LongTensor:
        """
        Sample some heads randomly from the current policy.
        """
        raise NotImplementedError()
    
    def update(
        self,
        heads: torch.LongTensor,
        rewards: torch.Tensor,
        on_policy: bool,
        heads_log_prob: Union[torch.Tensor, None] = None
    ) -> dict:
        """
        Update the underlying model based on the rewards.
        If the sampled heads are off-policy, the user must supply the log
        probabilities for the heads when the heads are sampled.
        """
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def load(self, path: str):
        raise NotImplementedError()
