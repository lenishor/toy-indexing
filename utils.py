import torch.nn as nn

from models import ToyTransformer


def weight_norm(model: nn.Module) -> float:
    """
    Returns the sum of the L^2 norm of the model parameters.
    """
    norm = 0.0
    for parameter in model.parameters():
        norm += parameter.norm().item()
    return norm


def qk_weight_norm(model: ToyTransformer) -> float:
    q_weight_norm = model.attention_head.query_projection.weight.norm().item()
    k_weight_norm = model.attention_head.key_projection.weight.norm().item()
    return q_weight_norm + k_weight_norm


def ov_weight_norm(model: ToyTransformer) -> float:
    return model.attention_head.output_projection.weight.norm().item()
