import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights

import argparse
from datetime import datetime
from enum import IntEnum
import logging
from typing import Union
from eval import eval_classification

import utils

class LayerType(IntEnum):
    Linear = 1              # Parameter:  out_features
    Conv = 2                # Parameters: out_channels, kernel_size, stride
    BatchNorm = 3           # Parameters: None
    LayerNorm = 4           # Parameters: None
    AvgPool = 5             # Parameters: kernel_size, stride
    MaxPool = 6             # Parameters: kernel_size, stride
    Upsampling = 7          # Parameter:  scale_factor
    Relu = 8                # Parameters: None
    Tanh = 9                # Parameters: None
    Sigmoid = 10            # Parameters: None
    SkipConnection = 11     # Parameters: starting_layer (use the output of this layer)
    End = 12

class Layer:
    def __init__(self, type: LayerType, parameters: tuple[int]):
        self.type = type
        self.parameters = parameters

def load_model(final_features: int, head_layers: list[Layer]) -> nn.Module:
    device = torch.cuda.current_device()
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    class Head(nn.Module):
        def __init__(self, in_features: int, head_layers: list[Layer]):
            super().__init__()
            self.in_features = in_features
            self.layers_metadata = head_layers
            self.layers = []

            curr_features = in_features
            for layer in self.layers_metadata:
                curr_layer = None
                if layer.type == LayerType.Linear:
                    curr_layer = nn.Linear(curr_features, layer.parameters[0]).to(device)
                    curr_features = layer.parameters[0]
                # TODO: complete this portion
                self.layers.append(curr_layer)

            # Add a linear layer at the end to make sure final dimensions match up
            final_layer = nn.Linear(curr_features, final_features).to(device)
            self.layers.append(final_layer)
        
        def forward(self, data):
            # Maintain list of outputs for skip connections
            outputs = [data]
            for i in range(len(self.layers_metadata) + 1):
                if self.layers[i] is not None:
                    data = self.layers[i](data)
                # Handle skip connections separately
                elif self.layers_metadata[i].type == LayerType.SkipConnection:
                    starting_layer = self.layers_metadata[i].parameters[0]
                    data = outputs[starting_layer+1] + outputs[-1] # TODO: dimension might not match
                outputs.append(data)
            return data

    model.fc = Head(model.fc.in_features, head_layers).to(device)
    if utils.get_world_size() > 1:
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
    batch_size: int = 256,
    epochs: int = 150,
    logdir: str = "./output/",
):
    device = torch.cuda.current_device()
    model.to(device)
    model.train()

    logger = logging.getLogger("train")
    logging.basicConfig(level=logging.INFO)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr)
        if epochs <= 100:
            logger.warning(
                "About to instantiate LR scheduler with milestone greater than" \
                "total number of epochs"
            )
        scheduler = MultiStepLR(
            optimizer,
            milestones=[75, 100],
            gamma=0.1,
        )

    if scheduler is None:
        # A constant LR scheduler that uses the LR of the optimizer
        scheduler = ConstantLR(
            optimizer,
            factor=1,
        )

    train_sampler = utils.get_sampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=4)
    writer = SummaryWriter(log_dir=logdir)

    current_iteration = 0
    for i in range(epochs):
        for data, gt_labels in dataloader:
            logits = model(data)
            loss = criterion(logits, gt_labels.to(device))
            if utils.is_main_process():
                writer.add_scalar("Training Loss", loss.item(), current_iteration)
                logger.info(
                    f"Epoch: {i+1}/{epochs}, Total iter: {current_iteration}, "
                    f"Training loss: {loss.item():.4f}, Batch size: {batch_size}, "
                    f"Learning rate: {scheduler.get_last_lr()}, "
                    f"Time: {datetime.now().ctime()}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_iteration += 1
        
        # Evaluate on validation set
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

def main(args):
    """
    Train and evaluate baseline model (i.e. only change dimension 
    of final linear layer).
    """
    # TODO: make final dim & criterion more flexible
    model = load_model(10, [])
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
    )
    # TODO: store model afterwards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar-10", type=str.lower)
    parser.add_argument("--num-gpus", default=1, type=int)

    parser.add_argument("--learning-rate", "-lr", default=0.1, type=float)
    parser.add_argument("--batch-size", "-b", default=256, type=int)
    parser.add_argument("--epochs", "-ep", default=150, type=int)

    parser.add_argument("--path-to-data", default="./data/", type=str)
    parser.add_argument("--logdir", default="./output/", type=str)

    args = parser.parse_args()
    utils.launch(main, args.num_gpus, (args, ))
