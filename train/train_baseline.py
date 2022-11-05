from operator import is_
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.models import resnet50, ResNet50_Weights

import argparse
from datetime import datetime
from enum import IntEnum
from functools import reduce
import logging
import os
import re
import time
from typing import Union

from eval.eval import eval_classification
import utils

class LayerType(IntEnum):
    Linear = 1              # Parameter:  out_features
    Conv = 2                # Parameters: out_channels, kernel_size, stride
    BatchNorm = 3           # Parameters: None
    LayerNorm = 4           # Parameters: None
    # Note: Using AdaptiveAvgPool2d / AdaptiveMaxPool2d for the two layers below
    AvgPool = 5             # Parameters: output_height, output_width
    MaxPool = 6             # Parameters: output_height, output_width
    Upsampling = 7          # Parameter:  scale_factor
    Relu = 8                # Parameters: None
    Tanh = 9                # Parameters: None
    Sigmoid = 10            # Parameters: None
    SkipConnection = 11     # Parameters: starting_layer (use the output of this layer)
    End = 12

_num_params_per_layer = {
    LayerType.Linear:           1,
    LayerType.Conv:             3,
    LayerType.BatchNorm:        0,
    LayerType.LayerNorm:        0,
    LayerType.AvgPool:          2,
    LayerType.MaxPool:          2,
    LayerType.Upsampling:       1,
    LayerType.Relu:             0,
    LayerType.Tanh:             0,
    LayerType.Sigmoid:          0,
    LayerType.SkipConnection:   1,
    LayerType.End:              0,
}

class Layer:
    def __init__(self, type: LayerType, parameters: tuple[int]):
        assert len(parameters) == _num_params_per_layer[type]
        self.type = type
        self.parameters = parameters

class HeadWoLinear(nn.Module):
    def __init__(self, in_size: list[int], head_layers: list[Layer]):
        super().__init__()
        self.in_size = in_size
        self.layers_metadata = head_layers
        self.num_layers = len(head_layers)
        # List of modules to log the parameters in PyTorch
        self.layers = nn.ModuleList()

        # Populate the layers
        curr_size = in_size
        for layer in self.layers_metadata:
            curr_layer : Union[nn.Module, None] = None
            if layer.type == LayerType.Linear:
                curr_layer = nn.Linear(curr_size[-1], layer.parameters[0])
            elif layer.type == LayerType.Conv:
                curr_layer = nn.Conv2d(
                    in_channels=curr_size[1],
                    out_channels=layer.parameters[0],
                    kernel_size=layer.parameters[1],
                    stride=layer.parameters[2]
                )
            elif layer.type == LayerType.AvgPool:
                curr_layer = nn.AdaptiveAvgPool2d(layer.parameters)
            # TODO: complete this portion

            # Keep track of current dimensions
            if curr_layer is not None:
                layer_info = summary(curr_layer, input_size=curr_size, verbose=0)
                curr_size = layer_info.summary_list[0].output_size
            self.layers.append(curr_layer)
        
        # Compute output features for the flatten layer after avgpool
        # (see PyTorch implementation of ResNet)
        self.output_features = reduce(lambda x, y: x * y, curr_size[1:], 1)
    
    def forward(self, data):
        # Maintain list of outputs for skip connections
        outputs = [data]
        for i in range(self.num_layers):
            if self.layers[i] is not None:
                data = self.layers[i](data)
            # Handle skip connections separately
            elif self.layers_metadata[i].type == LayerType.SkipConnection:
                starting_layer = self.layers_metadata[i].parameters[0]
                data = outputs[starting_layer+1] + outputs[-1] # TODO: dimension might not match
            outputs.append(data)
        return data

    @staticmethod
    def parse_head_wo_linear(head_repr: str) -> nn.Module:
        """
        See `train_custom_head.py`: `Head.parse_head` for more information.
        """
        layers = []
        for layer_repr in head_repr.split(";"):
            layer_repr = re.sub(r"\s", "", layer_repr)
            # Handle empty line
            if len(layer_repr) == 0:
                continue

            # Parse layer type and layer parameters
            layer_meta = layer_repr.split(":")
            assert len(layer_meta) == 2
            layer_type, layer_parameters_str = layer_meta
            layer_parameters = tuple(map(int, layer_parameters_str.split(",")))

            # Form layer
            layer = Layer(LayerType[layer_type], layer_parameters)
            layers.append(layer)

        # Hard-coded input size for AvgPool layer given 224x224 RGB images
        return HeadWoLinear([1, 2048, 7, 7], layers)

def load_model(
    num_classes: int,
    head_layers: str,
    from_checkpoint: str = "",
    freeze_backbone: bool = True,
    multi_gpu: bool = True,
) -> nn.Module:
    device = torch.cuda.current_device()
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

    # Freeze model backbone if necessary
    if freeze_backbone:
        for layer in model.children():
            for param in layer.parameters():
                param.requires_grad = False

    model.avgpool = HeadWoLinear.parse_head_wo_linear(head_layers).to(device)
    model.fc = nn.Linear(model.avgpool.output_features, num_classes).to(device)

    if from_checkpoint:
        model.load_state_dict(torch.load(from_checkpoint))

    # Use DDP to facilitate multi-GPU training
    if utils.get_world_size() > 1 and multi_gpu:
        model = DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device
        )
    return model

# TODO: reward & controller
def train(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    criterion,
    lr: float = 0.1,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None] = None,
    batch_size: int = 128,
    epochs: int = 300,
    logdir: str = "./output/",
    eval_between_epochs: bool = True,
):
    """
    Train the model from the given model.
    Note that the default hyperparameters here are for training from scratch,
    not for fine-tuning from a pretrained model.
    """
    device = torch.cuda.current_device()
    model.to(device)
    model.train()

    logger = logging.getLogger("train")
    # Make sure INFO-level logging is not ignored
    logging.basicConfig(level=logging.INFO)

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr,
            momentum=0.9,
            weight_decay=5e-4
        )
    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_sampler = utils.get_sampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=4)
    if utils.is_main_process():
        writer = SummaryWriter(log_dir=logdir)

    current_iteration = 0
    for i in range(epochs):
        data_start_time = time.time()
        for data, gt_labels in dataloader:
            data_time = time.time() - data_start_time
            
            inference_start_time = time.time()
            logits = model(data.to(device))
            inference_time = time.time() - inference_start_time

            loss = criterion(logits, gt_labels.to(device))
            if utils.is_main_process():
                writer.add_scalar("Data Time", data_time, current_iteration)
                writer.add_scalar("Inference Time", inference_time, current_iteration)
                writer.add_scalar(
                    "Learning Rate",
                    scheduler.get_last_lr()[0],
                    current_iteration
                )
                writer.add_scalar("Training Loss", loss.item(), current_iteration)
                
                logger.info(
                    f"Epoch: {i+1}/{epochs}, Total iter: {current_iteration}, "
                    f"Training loss: {loss.item():.4f}, Batch size: {batch_size}, "
                    f"Learning rate: {scheduler.get_last_lr()[0]:.4f}, "
                    f"Data time: {data_time:.4f} s, Inference time: {inference_time:.4f} s, "
                    f"Time: {datetime.now().ctime()}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_iteration += 1
            data_start_time = time.time()

        # Evaluate on validation set
        if eval_between_epochs or i == epochs - 1:
            model.eval()
            if utils.is_main_process():
                logger.info(f"Evaluating checkpoint after epoch {i+1}...")
            # TODO: support other CV tasks later on
            validation_acc = eval_classification(model, validation_dataset, batch_size)
            if utils.is_main_process():
                writer.add_scalar("Validation Accuracy", validation_acc, i)
                logger.info(
                    f"Epoch: {i+1}/{epochs}, Validation accuracy: {validation_acc:.3f}, "
                    f"Batch size: {batch_size}, Time: {datetime.now().ctime()}"
                )
            model.train()
        
        scheduler.step()

    return validation_acc

def main(args):
    """
    Train and evaluate baseline model with a user-determined head, changing the
    dimension of the final linear layer if necessary.
    """
    # TODO: make criterion more flexible

    # Make sure INFO-level logging is not ignored
    logging.basicConfig(level=logging.INFO)

    num_classes = utils.dataset_num_classes(args.dataset)
    model = load_model(
        num_classes,
        head_layers=args.head,
        freeze_backbone=args.freeze_backbone
    )
    train_dataset = utils.load_dataset(
        args.dataset,
        train=True,
        path_to_store=args.path_to_data
    )
    eval_dataset = utils.load_dataset(
        args.dataset,
        train=False,
        path_to_store=args.path_to_data
    )
    train(
        model,
        train_dataset,
        eval_dataset,
        nn.CrossEntropyLoss(),
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        logdir=args.logdir,
        eval_between_epochs=not args.no_eval_between_epochs,
    )

    if utils.is_main_process() and args.save_model:
        model_path = os.path.join(args.logdir, "final_model.pth")
        logger = logging.getLogger("train")
        logger.info(f"Saving model parameters to {model_path}...")

        # Retrieve underlying model in case model is DDP
        if utils.get_world_size() > 1:
            model = model.module

        # Move model to CPU to ensure it can be restored on any system
        model = model.cpu()
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar-10", type=str.lower)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--head", default="AvgPool: 1, 1", type=str)

    parser.add_argument("--learning-rate", "-lr", default=4e-3, type=float)
    parser.add_argument("--batch-size", "-b", default=128, type=int)
    parser.add_argument("--epochs", "-ep", default=100, type=int)
    parser.add_argument("--freeze-backbone", "-fb", action="store_true")
    parser.add_argument("--no-eval-between-epochs", "-nebe", action="store_true")

    parser.add_argument("--path-to-data", default="./data/", type=str)
    parser.add_argument("--save-model", "-sm", action="store_true")
    parser.add_argument("--logdir", default="./output/", type=str)

    args = parser.parse_args()
    utils.launch(main, args.num_gpus, (args, ))
