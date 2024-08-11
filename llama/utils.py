import torch
import time
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import os
import sys

def load_checkpoint(ckpt_path, weights_only=True):
    torch.cuda.nvtx.range_push("load_checkpoint")
    ckpt_start_time = time.time()
    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=weights_only)
    ckpt_end_time = time.time()
    print(f"Loading weights took {ckpt_end_time - ckpt_start_time} seconds")
    torch.cuda.nvtx.range_pop()
    return checkpoint

def model_memory_footprint(model):
    """
    Function to check the memory occupied by a large language model like LLaMA.
    
    Parameters:
    - model: PyTorch model (e.g., LLaMA).
    
    Returns:
    - memory_footprint: Memory occupied by the model in MB.
    """
    # Estimate memory by summing up the size of the model's parameters
    memory_footprint = sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 ** 2)
    print(f"Model memory footprint: {memory_footprint:.2f} MB")
    
    return memory_footprint

def print_model_architecture(model):
    """
    Function to print the architecture of a PyTorch model.
    
    Parameters:
    - model: PyTorch model.
    """
    print("Model Architecture:")
    print(model)
    
def check_tensors_on_device(model, device):
    """
    Function to check if all the tensors in a PyTorch model are on the specified device.
    
    Parameters:
    - model: PyTorch model.
    - device: Device to check the tensors on (e.g., "cuda" or "cpu").
    
    Returns:
    - tensors_on_device: Boolean indicating if all tensors are on the specified device.
    """
    all_params_on_device = True
    params_not_on_device = []
    for name, param in model.named_parameters():
        if device not in str(param.device):
            all_params_on_device = False
            params_not_on_device.append(name)

    if all_params_on_device:
        print("All parameters are on the correct device.")
    else:
        print("The following parameters are not on the correct device:")
        for param_name in params_not_on_device:
            print(f"- {param_name}")
    
    return all_params_on_device

def move_model_to_cuda(model):
    """
    Function to move a PyTorch model to the GPU (if available).
    
    Parameters:
    - model: PyTorch model.
    
    Returns:
    - model: PyTorch model moved to the GPU.
    """
    torch.cuda.nvtx.range_push("move_model_to_cuda")
    start_time_cuda = time.time()
    model = model.to("cuda")
    end_time_cuda = time.time()
    print(f"Model moved to CUDA in {end_time_cuda - start_time_cuda:.2f} seconds")
    torch.cuda.nvtx.range_pop()
    
    return model

def initial_setup(seed, model_parallel_size=None):
    torch.cuda.nvtx.range_push("initial_setup")
    # Clear CUDA memory before loading the model
    torch.cuda.empty_cache()
        
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(seed)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    
    torch.cuda.nvtx.range_pop()