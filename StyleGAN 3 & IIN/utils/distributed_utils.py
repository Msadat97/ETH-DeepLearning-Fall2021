"""
utils for working with pytorch distributed library. Useful when only using pytorch native distributed package to reduce the amount of required boilerplate codes.
"""

import functools
import ctypes
import pathlib

import torch
import torch.distributed as dist
import numpy as np

def init_process(rank, world_size, temp_dir, backend='nccl', **kwargs):
    """
    Initializer for pytorch distributed process group.
    """
    if dist.is_available():
        if dist.is_initialized():
            return
    
    init_file = pathlib.Path(temp_dir).joinpath(".torch_distributed_init")
    init_method = f"file://{init_file}"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", init_method=init_method, rank=rank, world_size=world_size, **kwargs
    )
    
    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL('libcudart.so')
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    else:
        return torch.device('cpu')

def get_rank():
    """Get rank of the thread.
    """
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def get_world_size():
    """Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


def is_local_master():
    return torch.cuda.current_device() == 0


def master_only(func):
    """Apply this function only to the master GPU."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


@master_only
def master_only_print(*args):
    """master-only print"""
    print(*args, flush=True)


def all_gather_tensor(tensor):
    """ gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
