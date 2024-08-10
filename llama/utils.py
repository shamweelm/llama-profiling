import torch
import time

def load_checkpoint(ckpt_path):
    ckpt_start_time = time.time()
    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    ckpt_end_time = time.time()
    print(f"Loading weights took {ckpt_end_time - ckpt_start_time} seconds")
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
    tensors_on_device = all(param.device == device for param in model.parameters())
    print(f"All tensors on device {device}: {tensors_on_device}")
    
    return tensors_on_device