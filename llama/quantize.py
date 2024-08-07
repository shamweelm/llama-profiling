import torch
from torch import nn
from functools import partial
from llama.model import Attention, RMSNorm, FeedForward, TransformerBlock, Transformer
from torchao.quantization.GPTQ import Int4WeightOnlyQuantizer
from torchao.quantization.quant_api import (
    quantize_,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    int8_weight_only
)

def int4_weight_only(model):
    
    
    groupsize = 64
    quantizer = Int4WeightOnlyQuantizer(
        groupsize,
    )
    quantized_model = quantizer.quantize(model)
    
    return quantized_model

torch_ao_quant_types = [
    "int8_dynamic_activation_int4_weight",
    "int8_dynamic_activation_int8_weight",
    "int8_dynamic_activation_int8_semi_sparse_weight",
    "int8_weight_only"
]

def torch_ao_quantize(model, quant_type):
    if quant_type == "int8_weight_only":
        quantize_(model, int8_weight_only())
    elif quant_type == "int8_dynamic_activation_int4_weight":
        quantize_(model, int8_dynamic_activation_int4_weight())
    elif quant_type == "int8_dynamic_activation_int8_weight":
        quantize_(model, int8_dynamic_activation_int8_weight())
    elif quant_type == "int8_dynamic_activation_int8_semi_sparse_weight":
        quantize_(model, int8_dynamic_activation_int8_semi_sparse_weight())
    else:
        raise ValueError(f"Invalid quant_type: {quant_type}")


def quantize_model(model, quant_type):
    if quant_type == "int4_weight_only":
        model = int4_weight_only(model)
    elif quant_type in torch_ao_quant_types:
        torch_ao_quantize(model, quant_type)
    else:
        raise ValueError(f"Invalid quant_type: {quant_type}")
        
    return model