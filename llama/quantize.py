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


class QuantizedInt8LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        self.register_buffer(
            "weight",
            torch.randint(-128, 127, (out_features, in_features)).to(torch.int8),
        )

        self.register_buffer("scale", torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weight):
        # Clone the weight and outcast it to fp32 which is necessary to calculate the scale as both types must be in fp32
        weight_f32 = weight.clone().to(torch.float32)

        # calculating the min and max of int-8 quantized range. qmin=-128, qmax=127
        Qmin = torch.iinfo(torch.int8).min
        Qmax = torch.iinfo(torch.int8).max

        # calculating per channel scale
        # In per channel scale, you'll be calculating the scale for every row. So, you'll store the scale in a tensor in this case.)
        # In per tensor scale, you'll calculate one scale for entire tensor. Per channel will be more accurate but take more memory footprint as it has to store more scale value.
        # weight_f32.abs().max(dim=-1).values -> this give the max-value for original weight value range in fp32.
        scale = weight_f32.abs().max(dim=-1).values / 127
        scale = scale.to(weight.dtype)

        # This gives the quantized weight value for the given weight tensor.
        # This formula was derived from symmetric quantization. please read the link I've shared above if you want to learn in detail.
        quantized_weight = torch.clamp(
            torch.round(weight / scale.unsqueeze(1)), Qmin, Qmax
        ).to(torch.int8)

        self.weight = quantized_weight
        self.scale = scale

    def forward(self, input):
        output = F.linear(input, self.weight.to(input.dtype)) * self.scale
        if self.bias is not None:
            output = output + self.bias

        return output


def replace_linearlayer_custom_qint8(
    base_model, quantizer_class, exclude_list, quantized=True
):
    # Convert the exclude list to a set for faster lookup
    exclude_set = set(exclude_list)

    # Assume model is on GPU, get device
    device = next(base_model.parameters()).device

    # Iterate over named modules directly for in-place replacement
    for name, module in base_model.named_modules():
        # Skip layers in the exclude list
        if any(excl in name for excl in exclude_set):
            continue

        # Only process nn.Linear layers
        if isinstance(module, nn.Linear):
            layer_name = name.split(".")[-1]
            parent_module = base_model
            sub_names = name.split(".")
            for sub_name in sub_names[:-1]:
                parent_module = getattr(parent_module, sub_name)

            # Fetch module parameters
            old_weight = module.weight.data
            old_bias = module.bias.data if module.bias is not None else None
            in_features = module.in_features
            out_features = module.out_features

            # Initialize the quantizer layer directly on GPU
            quantizer_layer = quantizer_class(
                in_features, out_features, old_bias is not None, old_weight.dtype
            ).to(device)

            # Quantize weights directly on GPU
            if quantized:
                quantizer_layer.quantize(old_weight)

            # Restore bias if it exists
            if old_bias is not None:
                quantizer_layer.bias.data.copy_(old_bias)

            # Replace the original Linear layer with the quantizer layer in-place
            setattr(parent_module, layer_name, quantizer_layer)

            # Remove old references to free memory
            del old_weight, old_bias


class QuantizedInt8LinearDynamicActivationLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        # Static int8 weight quantization
        self.register_buffer(
            "weight",
            torch.randint(-128, 127, (out_features, in_features)).to(torch.int8),
        )

        self.register_buffer("scale", torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weight):
        # Clone the weight and cast it to fp32 for scale calculation
        weight_f32 = weight.clone().to(torch.float32)

        # Calculate the min and max of the int8 quantized range
        Qmin = torch.iinfo(torch.int8).min
        Qmax = torch.iinfo(torch.int8).max

        # Calculate per-channel scale (one scale per row)
        scale = weight_f32.abs().max(dim=-1).values / 127
        scale = scale.to(weight.dtype)

        # Quantize the weight tensor
        quantized_weight = torch.clamp(
            torch.round(weight / scale.unsqueeze(1)), Qmin, Qmax
        ).to(torch.int8)

        self.weight = quantized_weight
        self.scale = scale

    def dynamic_quantize_activations(self, input):
        # Calculate the min and max of the input activations dynamically
        activation_min = input.min()
        activation_max = input.max()

        # Optionally expand the dynamic range slightly to reduce clipping
        activation_min = activation_min * 1.1  # Slightly lower the minimum
        activation_max = activation_max * 1.1  # Slightly increase the maximum

        # Calculate the scale and zero-point dynamically
        scale = (activation_max - activation_min) / 255.0
        zero_point = torch.round(-activation_min / scale).to(torch.int8)

        # Quantize the input activations
        quantized_input = torch.clamp(
            torch.round(input / scale) + zero_point, -128, 127
        ).to(torch.int8)

        return quantized_input, scale, zero_point

    def forward(self, input):
        # Dynamically quantize the input activations
        quantized_input, scale, zero_point = self.dynamic_quantize_activations(input)

        # Perform the linear operation with the input tensor and matching weight dtype
        output = F.linear(quantized_input.to(input.dtype), self.weight.to(input.dtype)) * self.scale.to(input.dtype)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        return output


def replace_linearlayer_with_static_weight_and_dynamic_activation(
    base_model, quantizer_class, exclude_list, quantized=True
):

    # Convert the exclude list to a set for faster lookup
    exclude_set = set(exclude_list)

    # Assume model is on GPU, get device
    device = next(base_model.parameters()).device

    # Iterate over named modules directly for in-place replacement
    for name, module in base_model.named_modules():
        # Skip layers in the exclude list
        if any(excl in name for excl in exclude_set):
            continue

        # Only process nn.Linear layers
        if isinstance(module, nn.Linear):
            layer_name = name.split(".")[-1]
            parent_module = base_model
            sub_names = name.split(".")
            for sub_name in sub_names[:-1]:
                parent_module = getattr(parent_module, sub_name)

            # Fetch module parameters
            old_weight = module.weight.data
            old_bias = module.bias.data if module.bias is not None else None
            in_features = module.in_features
            out_features = module.out_features

            # Initialize the quantizer layer directly on GPU
            quantizer_layer = quantizer_class(
                in_features, out_features, old_bias is not None, old_weight.dtype
            ).to(device)

            # Quantize weights directly on GPU
            if quantized:
                quantizer_layer.quantize(old_weight)

            # Restore bias if it exists
            if old_bias is not None:
                quantizer_layer.bias.data.copy_(old_bias)

            # Replace the original Linear layer with the quantizer layer in-place
            setattr(parent_module, layer_name, quantizer_layer)

            # Remove old references to free memory
            del old_weight, old_bias


class Quantizer:
    def __init__(self, model: nn.Module, quantization_type: str, ckpt_dir: str):
        self.model = model
        self.quantization_type = quantization_type
        self.ckpt_dir = ckpt_dir
        self.quantized_path = (
            Path(ckpt_dir) / f"quantized_model_{quantization_type}.pth"
        )

    def quantize_based_on_type(self, custom_quantize=True):
        torch.cuda.nvtx.range_push(f"quantize_based_on_type_{self.quantization_type}") 
        start_time = time.perf_counter()
        if self.quantization_type == "torch_ao_int4_wo":
            group_size = 64
            quantize_(self.model, int4_weight_only(group_size=group_size))
        elif self.quantization_type == "torch_ao_int8_wo":
            quantize_(self.model, int8_weight_only())
        elif self.quantization_type == "custom_int8_ll_wt_no_output":
            replace_linearlayer_custom_qint8(
                self.model,
                QuantizedInt8LinearLayer,
                exclude_list=["output"],
                quantized=custom_quantize,
            )
        elif self.quantization_type == "custom_int8_ll_wt_no_output_dyn_act":
            replace_linearlayer_with_static_weight_and_dynamic_activation(
                self.model,
                QuantizedInt8LinearDynamicActivationLayer,
                exclude_list=["output"],
                quantized=custom_quantize,
            )
        elif self.quantization_type == "custom_int8_ll_wt":
            replace_linearlayer_custom_qint8(
                self.model,
                QuantizedInt8LinearLayer,
                exclude_list=[],
                quantized=custom_quantize,
            )
        elif self.quantization_type == "custom_int8_ll_wt_dyn_act":
            replace_linearlayer_with_static_weight_and_dynamic_activation(
                self.model,
                QuantizedInt8LinearDynamicActivationLayer,
                exclude_list=[],
                quantized=custom_quantize,
            )
        end_time = time.perf_counter()
        print(f"Quantization took {end_time - start_time} seconds")
        torch.cuda.nvtx.range_pop()

    def load_quantized_model(self, custom_quantize=False):
        torch.cuda.nvtx.range_push("load_quantized_model")
        print(f"Quantized model already exists at {self.quantized_path}")
        self.quantize_based_on_type(custom_quantize=custom_quantize)

        torch.cuda.nvtx.range_push("load_quantized_weights")
        # Get size of the quantized model
        quantized_model_size = self.quantized_path.stat().st_size / (1024**2)
        print(f"Quantized model size: {quantized_model_size:.2f} MB")

        # Load the quantized model
        checkpoint = load_checkpoint(self.quantized_path, weights_only=False)
        torch.cuda.nvtx.range_push("model_load_state_dict")
        self.model.load_state_dict(checkpoint, strict=False, assign=True)
        torch.cuda.nvtx.range_pop()
        del checkpoint
        torch.cuda.nvtx.range_pop()
        return self.model

    def save_quantized_model(self, model):
        torch.cuda.nvtx.range_push("save_quantized_model")
        torch.save(model.state_dict(), self.quantized_path)
        print(f"Quantized model saved to {self.quantized_path}")

        # Get size of the quantized model
        quantized_model_size = self.quantized_path.stat().st_size / (1024**2)
        print(f"Quantized model size: {quantized_model_size:.2f} MB")
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
        quantized_path = (
            Path(self.ckpt_dir) / f"quantized_model_{self.quantization_type}.pth"
        )

        if quantized_path.exists():
            return self.load_quantized_model(custom_quantize=False)
        else:
            ckpt_path = [
                ckpt
                for ckpt in checkpoints
                if ckpt.name.endswith("consolidated.00.pth")
            ][0]
            checkpoint = load_checkpoint(ckpt_path)
            torch.cuda.nvtx.range_push("model_load_state_dict")
            self.model.load_state_dict(checkpoint, strict=False, assign=True)
            torch.cuda.nvtx.range_pop()
            # Quantize the model based on the specified quantization type
            self.quantize_based_on_type(custom_quantize=True)
            # Update self.model
            self.save_quantized_model(self.model)

        # Clear CUDA memory after loading the model
        torch.cuda.empty_cache()

        return self.model
