import torch
import torch.nn as nn
import torch.nn.functional as F

# Define QuantizedLinearLayer class
class QuantizedLinearLayer(nn.Module):
    # As our target is to replace the linear layer from the base model. We must use the same parameters 
    # such as in_features, out_features, bias = True, dtype=torch.float32, The dtype is a type of bias
    def __init__(self, in_features, out_features, bias = True, dtype=torch.float32):
        super().__init__()

        # Note that we're using self.register_buffer to store parameter variable value. This is because if we use nn.Parameter, the network will start calculating the gradient. 
        # We don't want that as we're not using this for training.
        # weight will be initialized randomly between (-128, 127) which is a range of signed int-8        
        self.register_buffer("weight", torch.randint(-128, 127, (out_features, in_features)).to(torch.int8))

        # scale will have dimension and data type same as the output as this will be multiplied to the output of linear layer
        self.register_buffer("scale", torch.randn((out_features), dtype=dtype))

        # bias is an optional parameter, so we only add it if is not none. 
        # bias dimension is (1, out_features) as it can later broadcasted during addition. 
        if bias:          
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None

  # 8-bit quantization function
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
        scale = weight_f32.abs().max(dim=-1).values/127
        scale = scale.to(weight.dtype)

        # This gives the quantized weight value for the given weight tensor. 
        # This formula was derived from symmetric quantization. please read the link I've shared above if you want to learn in detail.
        quantized_weight = torch.clamp(torch.round(weight/scale.unsqueeze(1)), Qmin, Qmax).to(torch.int8)

        self.weight = quantized_weight
        self.scale = scale
    
    def forward(self, input):
        # This gives the output the same way as the linear function in the base model. 
        # The only difference is that the weight value is now the quantized weight value. 
        # Hence, this gives less processing by faster calculation and less memory utilization.

        output = F.linear(input, self.weight.to(input.dtype)) * self.scale
        if self.bias is not None:
            output = output + self.bias
        return output
    
    

def replace_linearlayer(base_model, quantizer_class, exclude_list, quantized=True):

    # Finding only the instance of base_model which has the linear layer
    # Also we have to make sure to exclude those linearlayer that are in the exclude list.
    for name, child in base_model.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
            old_bias = child.bias
            old_weight = child.weight
            in_features = child.in_features
            out_features = child.out_features

          # This is the stage where we'll initialize the quantizer class with the in_features, out_features, bias and dtype.
          # The base_model parameters values are given to the quantizer class parameters.
            quantizer_layer = quantizer_class(in_features, out_features, old_bias is not None, old_weight.dtype)

          # After the quantizer class is initialized, The replacement takes place as below.
            setattr(base_model, name, quantizer_layer)

          # Now that after replacement, base_model linear layer is now a quantizer layer.
          # We can now call the quantize_layers quantize function to quantize the old_weights of FP16 new quantized weights of int8 type.
            if quantized:
                getattr(base_model, name).quantize(old_weight)

          # If bias is not none, we'll also update bias with the base model bias value
            if old_bias is not None:
                getattr(base_model, name).bias = old_bias

        # If the base model child instance has further sub-components with linear layers, we'll have to quantize them by calling the replace_linear_layer function with the child as base_model now.
        # This will replace all the linear layers with quantized layers that are under the child sub-section.
        else:
            replace_linearlayer(child, quantizer_class, exclude_list, quantized=quantized)
            

def get_memory_footprint(model, input_size=(1, 2048), device='cuda'):
    """
    Function to check the memory occupied by a large language model like LLaMA.
    
    Parameters:
    - model: PyTorch model (e.g., LLaMA).
    - input_size: Tuple, default is (1, 1024). The input size to pass through the model (batch_size, sequence_length).
    - device: String, 'cuda' or 'cpu'. The device on which the model is loaded.

    Returns:
    - memory_footprint: Memory occupied by the model in MB.
    """
    model.to(device)
    
    # Switch to evaluation mode
    model.eval()
    
    # Create a dummy input of the specified size
    dummy_input = torch.randint(0, model.vocab_size, input_size).to(device)
    
    if device == 'cuda':
        # Clear the cache and reset memory trackers
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Forward pass
        with torch.no_grad():
            model(dummy_input)

        # Get memory usage in bytes and convert to MB
        memory_footprint = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    else:
        # Estimate memory by summing up the size of the model's parameters
        memory_footprint = sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 ** 2)
    
    return memory_footprint