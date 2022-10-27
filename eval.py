import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils

def eval_classification(
    model: nn.Module,
    eval_dataset: Dataset,
    batch_size: int = 128,
) -> float:
    device = torch.cuda.current_device()
    model.to(device)
    model.eval()

    test_sampler = utils.get_sampler(eval_dataset)
    dataloader = DataLoader(eval_dataset, batch_size, sampler=test_sampler, num_workers=4)
    correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for data, gt_labels in dataloader:
            gt_labels = gt_labels.to(device)
            logits = model(data)
            pred_labels = torch.argmax(logits, dim=-1)
            correct_preds += (pred_labels == gt_labels).int().sum().item()
            total_samples += gt_labels.size(0)

    if utils.get_world_size() > 1:
        stats = torch.tensor([correct_preds, total_samples], dtype=torch.int64).to(device)
        tensor_list = [
            torch.zeros((2,), dtype=torch.int64).to(device) \
            for _ in range(utils.get_world_size())
        ]
        dist.all_gather(tensor_list, stats)
        correct_preds, total_samples = torch.vstack(tensor_list).sum(dim=0).cpu().numpy()

    return correct_preds / total_samples
