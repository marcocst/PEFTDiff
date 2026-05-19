import torch
from diffusers.loaders import AttnProcsLayers
from typing import Optional

from .attention_processor import TTMLoRAAttnProcessor

TENSOR_SHAPE_DICT = {
    '320': [4, 8, 10],
    '640': [8, 8, 10],
    '768': [8, 8, 12],
    '1280': [8, 10, 16],
    '2048': [8, 16, 16]
}


def combine_tensor_shape_dict(tensor_shape_dict: Optional[dict]):
    new_tensor_shape_dict = TENSOR_SHAPE_DICT.copy()
    if tensor_shape_dict is not None:
        for k, v in tensor_shape_dict.items():
            new_tensor_shape_dict[k] = v
    return new_tensor_shape_dict


def register_adaptation(
        model,
        rank: int = 4,
        tensor_shape_dict: Optional[dict] = None,
        transform_rank: int = 1,
        alpha: float = 1.0,
        addition_part: str = 'lora',
        use_rslora: bool = True
    ):
    tensor_shape_dict = combine_tensor_shape_dict(tensor_shape_dict)

    td_attn_procs = {}
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]

        td_attn_procs[name] = TTMLoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
            hidden_tensor_shape=tensor_shape_dict[str(hidden_size)],
            cross_tensor_shape=None if cross_attention_dim is None else tensor_shape_dict[str(cross_attention_dim)],
            rank=rank, transform_rank=transform_rank,
            alpha=alpha, addition_part=addition_part, use_rslora=use_rslora
        )

    model.set_attn_processor(td_attn_procs)
    td_layers = AttnProcsLayers(model.attn_processors)
    return model, td_layers
