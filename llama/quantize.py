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
from llama.utils import load_checkpoint


class Quantizer:
    def __init__(self, model: nn.Module, quantization_type: str, ckpt_dir: str):
        self.model = model
        self.quantization_type = quantization_type
        self.ckpt_dir = ckpt_dir
        self.quantized_path = Path(ckpt_dir) / f"quantized_model_{quantization_type}.pth"
    
    def quantize_based_on_type(self):
        torch.cuda.nvtx.range_push("quantize_based_on_type")
        start_time = time.perf_counter()
        if self.quantization_type == "int4_weight_only":
            group_size = 32
            quantize_(self.model, int4_weight_only(group_size=group_size))
        elif self.quantization_type == "int8_weight_only":
            quantize_(self.model, int8_weight_only())
        end_time = time.perf_counter()
        print(f"Quantization took {end_time - start_time} seconds")
        torch.cuda.nvtx.range_pop()
        return self.model
    
    def load_quantized_model(self):
        torch.cuda.nvtx.range_push("load_quantized_model")
        print(f"Quantized model already exists at {self.quantized_path}")
        model = self.quantize_based_on_type()
        
        torch.cuda.nvtx.range_push("load_quantized_weights")
        # Load the quantized model
        checkpoint = load_checkpoint(self.quantized_path)
        model.load_state_dict(checkpoint, strict=False, assign=True)
        torch.cuda.nvtx.range_pop()
        return model
    
    def save_quantized_model(self, model):
        torch.cuda.nvtx.range_push("save_quantized_model")
        torch.save(model.state_dict(), self.quantized_path)
        print(f"Quantized model saved to {self.quantized_path}")
        torch.cuda.nvtx.range_pop()
        
    def quantize(
        self,
    ) -> nn.Module:
        """
        Function to quantize a PyTorch model based on the specified quantization type.
        
        Returns:
        - model: Quantized PyTorch model.
        """
        torch.cuda.nvtx.range_push("quantize_model")
    
        checkpoints = sorted(Path(self.ckpt_dir).glob("*.pth"))
        quantized_path = Path(self.ckpt_dir) / f"quantized_model_{self.quantization_type}.pth"
        
        if quantized_path.exists():
            model = self.load_quantized_model()
        else:
            ckpt_path = [ckpt for ckpt in checkpoints if ckpt.name.endswith("consolidated.00.pth")][0]
            checkpoint = load_checkpoint(ckpt_path)
            model.load_state_dict(checkpoint, strict=False, assign=True)
            model = self.quantize_based_on_type()
            # Update self.model
            self.save_quantized_model(model)
            
        # Clear CUDA memory after loading the model
        torch.cuda.empty_cache()
        
        return model
