import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from datetime import timedelta
import logging
import os
import socket

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dataset(name: str,
                 train: bool = True,
                 path_to_store: str = "./data/",
                 transform = None) -> Dataset:
    if is_main_process():
        logger = logging.getLogger("utils")
        logger.info(f"Attempting to load dataset {name}...")

    # Required transformations for pretrained Resnet-50
    mean, std = dataset_mean_std(name)
    if transform is None:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize(256),
            T.RandomCrop(224),
            T.Normalize(mean, std),
        ])
        # Additional augmentations in training setting
        if train:
            transform = T.Compose([
                T.GaussianBlur((5, 5)),
                T.ColorJitter(brightness=.5, hue=.3),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                transform,
            ])

    if name == "cifar-10":
        return CIFAR10(path_to_store, train, transform, download=True)
    elif name == "cifar-100":
        return CIFAR100(path_to_store, train, transform, download=True)
    else:
        logger = logging.getLogger("utils")
        logger.error("Dataset %s not yet supported!" % name)

def dataset_mean_std(name: str):
    if name == "cifar-10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif name == "cifar-100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    else:
        logger = logging.getLogger("utils")
        logger.error("Dataset %s not yet supported!" % name)
    return mean, std

def dataset_num_classes(name: str) -> int:
    if name == "cifar-10":
        return 10
    elif name == "cifar-100":
        return 100
    else:
        logger = logging.getLogger("utils")
        logger.error("Dataset %s not yet supported!" % name)

def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # This forces socket to find a free port for me on all interfaces
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # There is a chance that this port will still be occupied
    # (synchronization issue)
    return port

DEFAULT_TIMEOUT = timedelta(minutes=10)

def launch(
    main_func,
    num_gpus: int,
    args: tuple = (),
    timeout=DEFAULT_TIMEOUT,
):
    if num_gpus > 1:
        port = _find_free_port()
        dist_url = "tcp://127.0.0.1:%d" % port
        mp.spawn(
            _dist_worker,
            args=(
                main_func,
                num_gpus,
                dist_url,
                args,
                timeout,
            ),
            nprocs=num_gpus,
        )
    else:
        main_func(*args)

def _dist_worker(
    local_rank,
    main_func,
    num_gpus,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert torch.cuda.is_available(), "CUDA unavailable!"
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            timeout=timeout,
            world_size=num_gpus,
            rank=local_rank,
        )
    except Exception as e:
        logger = logging.getLogger("utils")
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    
    assert num_gpus <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Synchronize all GPUs
    dist.barrier(device_ids=[local_rank])

    main_func(*args)

# DDP utilities
def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()

def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_sampler(dataset: Dataset) -> Sampler:
    # Special handling since DistributedSampler cannot be used when process group is
    # not initialized
    if get_world_size() == 1:
        return RandomSampler(dataset)
    else:
        return DistributedSampler(dataset, shuffle=True)
