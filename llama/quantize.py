from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import tqdm
import torch
import torch.nn as nn
import copy
from torchao.quantization.quant_api import (
    quantize_,
    int4_weight_only,
    int8_weight_only,
)
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
import time

# Define QuantizedLinearLayer class
def quantize_model(model, quantization_type, ckpt_dir):
    
    quantized_path = Path(ckpt_dir) / f"quantized_model_{quantization_type}.pth"
    
    if quantized_path.exists():
        print(f"Quantized model found at {quantized_path}")
        return model
    
    start_time = time.perf_counter()
    if quantized_path.exists():
        model.load_state_dict(torch.load(quantized_path))
        return model
    
    if quantization_type == "int4_weight_only":
        group_size = 32
        quantize_(model, int4_weight_only(group_size=group_size))
    elif quantization_type == "int8_weight_only":
        quantize_(model, int8_weight_only())
    
    end_time = time.perf_counter()
    print(f"Quantization took {end_time - start_time} seconds")
        
    # Save the quantized model in the checkpoint directory
    torch.save(model.state_dict(), Path(ckpt_dir) / f"quantized_model_{quantization_type}.pth")
    print(f"Quantized model saved to {quantized_path}")
    
    return model


def get_memory_footprint(model):
    """
    Function to check the memory occupied by a large language model like LLaMA.
    
    Parameters:
    - model: PyTorch model (e.g., LLaMA).
    
    Returns:
    - memory_footprint: Memory occupied by the model in MB.
    """
    # Estimate memory by summing up the size of the model's parameters
    memory_footprint = sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 ** 2)
    
    return memory_footprint