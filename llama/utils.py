import torch
import time

def load_checkpoint(ckpt_path, weights_only=True):
    ckpt_start_time = time.time()
    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=weights_only)
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
    params_not_on_device = []
    for name, param in model.named_parameters():
        if param.device != device:
            all_params_on_device = False
            params_not_on_device.append(name)

    if all_params_on_device:
        print("All parameters are on the correct device.")
    else:
        print("The following parameters are not on the correct device:")
        for param_name in params_not_on_device:
            print(f"- {param_name}")
    
    return all_params_on_device