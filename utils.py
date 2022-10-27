import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet50_Weights

from datetime import timedelta
import logging
import socket

def load_dataset(name: str,
                 train: bool = True,
                 path_to_store: str = "./data") -> torch.utils.data.Dataset:
    # Required transformations for pretrained Resnet
    transform = ResNet50_Weights.DEFAULT.transforms()

    if name == "cifar-10":
        return CIFAR10(path_to_store, train, transform, download=True)
    elif name == "cifar-100":
        return CIFAR100(path_to_store, train, transform, download=True)
    else:
        logger = logging.getLogger("utils")
        logger.error("Dataset %s not yet supported!" % name)

# TODO: add dataset to final dim utility

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
