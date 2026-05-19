# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union, List
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


def ttm_to_matrix(cores: List[torch.Tensor], flatten: bool = True):
    out = cores[0]
    row = out.shape[0]
    col = out.shape[1]
    for d in range(1, len(cores)):
        out = torch.einsum('...mn, ...ijnt->...ijmt', out, cores[d])
        out = torch.moveaxis(out, 2 * d, d)
        row *= cores[d].shape[0]
        col *= cores[d].shape[1]

    out = torch.einsum('...ii-> ...', out)
    if flatten:
        return out.reshape(row, col)
    else:
        return out


def ttv_to_matrix(cores: List[torch.Tensor], flatten: bool = True):
    out = cores[0]
    for d in range(1, len(cores)):
        out = torch.einsum('...mn, ...int->...imt', out, cores[d])

    out = torch.einsum('...ii-> ...', out)
    if flatten:
        return out.reshape(-1)
    else:
        return out


def seq_kronecker(cores: List[torch.Tensor], flatten: bool = True):
    out = cores[0]
    row = [out.shape[0]]
    col = [out.shape[1]]
    for d in range(1, len(cores)):
        out = torch.kron(out, cores[d])
        row.append(cores[d].shape[0])
        col.append(cores[d].shape[1])

    if flatten:
        return out
    else:
        shape = [item for pair in zip(row, col) for item in pair]
        return out.reshape(shape)  # TODO Is this permutation right?


class Attention(nn.Module):
    r"""
    A cross attention layer.
    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5 if scale_qk else 1.0

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
            self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        is_lora = hasattr(self, "processor") and isinstance(
            self.processor, (LoRAAttnProcessor, LoRAXFormersAttnProcessor)
        )

        if use_memory_efficient_attention_xformers:
            if self.added_kv_proj_dim is not None:
                # TODO(Anton, Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                raise NotImplementedError(
                    "Memory efficient attention with `xformers` is currently not supported when"
                    " `self.added_kv_proj_dim` is defined."
                )
            elif not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            if is_lora:
                processor = LoRAXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                processor.to(self.processor.to_q_lora.up.weight.device)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_lora:
                processor = LoRAAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                )
                processor.load_state_dict(self.processor.state_dict())
                processor.to(self.processor.to_q_lora.up.weight.device)
            else:
                processor = AttnProcessor()

        self.set_processor(processor)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            processor = AttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None, out_dim=3):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.0.15",
                (
                    "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect"
                    " attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to"
                    " `prepare_attention_mask` when preparing the attention_mask."
                ),
            )
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states):
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states


class TTMLoRALinearLayer(nn.Module):
    def __init__(
            self, in_features, out_features,
            in_tensor_shape: list, out_tensor_shape: Optional[list],  # NOTE Maybe we should also consider add transform for in features
            rank=10, transform_rank=1,
            alpha=1., addition_part='lora',
            use_rslora=True
        ):
        super(TTMLoRALinearLayer, self).__init__()

        self.in_features=in_features
        self.out_features=out_features
        self.rank = rank
        self.transform_rank = transform_rank
        self.alpha = alpha
        self.addition_part = addition_part.lower()
        self.use_rslora = use_rslora
        self.in_tensor_shape = in_tensor_shape
        self.out_tensor_shape = out_tensor_shape

        # check shapes
        assert math.prod(in_tensor_shape) == in_features
        assert math.prod(out_tensor_shape) == out_features

        assert self.addition_part in ['lora', 'ttm', 'ttv'], f"Method {addition_part} not implemented! Use 'lora', 'ttm' or 'ttv'"

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))

        # initialize TTM transform
        if transform_rank > 0:
            assert in_tensor_shape is not None
            transform_core = []
            for d in range(len(in_tensor_shape)):
                if self.transform_rank == 1:  # we use Kronecker factors
                    transform_core.append(nn.Parameter(torch.eye(in_tensor_shape[d]), requires_grad=True))
                else:  # we use TTM factors
                    transform_core.append(
                        nn.Parameter(
                            torch.stack([torch.eye(in_tensor_shape[d]) for _ in range(transform_rank ** 2)], -1).view(
                                in_tensor_shape[d], in_tensor_shape[d], transform_rank, transform_rank
                            ) / transform_rank, requires_grad=True
                        )
                    )
            self.transform_core = nn.ParameterList(transform_core)
        else:
            self.transform_core = None

        # initialize additional part
        if self.addition_part == 'lora':
            self.lora_B = nn.Parameter(torch.empty(out_features, rank), requires_grad=True)
            self.lora_A = nn.Parameter(torch.empty(rank, in_features), requires_grad=True)
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5.))
        elif self.addition_part == 'ttv':  # use TT-Vector
            ttv_core_in = []
            ttv_core_out = []
            for d in range(len(in_tensor_shape)):
                init_fun = lambda x: nn.init.normal_(x, std=1./math.sqrt(in_tensor_shape[d]))
                ttv_core_in.append(
                    nn.Parameter(torch.empty(in_tensor_shape[d], rank, rank), requires_grad=True)
                )
                init_fun(ttv_core_in[-1])
            for d in range(len(out_tensor_shape)):
                if d == 0:
                    init_fun = nn.init.zeros_
                else:
                    init_fun = lambda x: nn.init.normal_(x, std=math.sqrt(out_tensor_shape[d])/rank ** 2)
                ttv_core_out.append(
                    nn.Parameter(torch.empty(out_tensor_shape[d], rank, rank), requires_grad=True)
                )
                init_fun(ttv_core_out[-1])
            self.ttv_core = nn.ParameterList([*ttv_core_out, *ttv_core_in])
        else:  # use TT-Matrix
            assert len(in_tensor_shape) == len(out_tensor_shape)
            ttm_core = []
            for d in range(len(in_tensor_shape)):
                if d == 0:
                    init_fun = nn.init.zeros_
                else:
                    init_fun = lambda x: nn.init.kaiming_normal_(x, a=math.sqrt(5.))
                ttm_core.append(
                    nn.Parameter(torch.empty(out_tensor_shape[d], in_tensor_shape[d], rank, rank), requires_grad=True)
                )
                init_fun(ttm_core[-1])
            self.ttm_core = nn.ParameterList(ttm_core)

        self.fix_filt_shape = [in_features, out_features]

    def forward(self, attn, x):
        orig_dtype = x.dtype
        # dtype = self.transform_core[0].dtype
        if self.addition_part == 'lora':
            dtype = self.lora_B.dtype
        else:
            dtype = self.ttv_core[0].dtype

        # fix filter
        weight = attn.weight.data

        # apply transform
        if self.use_rslora:
            scaling = self.alpha / math.sqrt(self.rank)
            # TODO This should be justified for tensors
        else:
            scaling = self.alpha / self.rank
        if self.transform_rank == 1:
            transform = seq_kronecker(self.transform_core)
        elif self.transform_rank <= 0:
            transform = None
        else:
            transform = ttm_to_matrix(self.transform_core)

        if self.addition_part == 'lora':
            delta_weight = torch.matmul(self.lora_B, self.lora_A)
        elif self.addition_part == 'ttv':
            delta_weight = ttv_to_matrix(self.ttv_core, flatten=False).view(self.out_features, self.in_features)
        else:
            delta_weight = ttm_to_matrix(self.ttm_core, flatten=True)
        if transform is not None:
            weight = torch.matmul(weight.to(dtype), transform) + scaling * delta_weight
        else:
            weight = weight.to(dtype) + scaling * delta_weight

        # Apply the trainable identity matrix
        bias_term = attn.bias.data if attn.bias is not None else None
        if bias_term is not None:
            bias_term = bias_term.to(orig_dtype)

        out = nn.functional.linear(input=x.to(orig_dtype), weight=weight.to(orig_dtype), bias=bias_term)

        return out


class TTMLoRAAttnProcessor(nn.Module):
    def __init__(
            self, hidden_size, cross_attention_dim=None, 
            hidden_tensor_shape: list = None, cross_tensor_shape: Optional[list] = None,
            rank=10, transform_rank=1,
            alpha=1.0, addition_part='lora', use_rslora=True
        ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.hidden_tensor_shape = hidden_tensor_shape
        self.cross_tensor_shape = cross_tensor_shape
        self.rank = rank
        self.transform_rank = transform_rank
        self.alpha = alpha
        self.addition_part = addition_part
        self.use_rslora = use_rslora
        
        self.to_q_oft = TTMLoRALinearLayer(
            hidden_size, hidden_size,
            in_tensor_shape=hidden_tensor_shape, out_tensor_shape=hidden_tensor_shape,
            rank=rank, transform_rank=transform_rank,
            alpha=alpha, use_rslora=use_rslora, addition_part=addition_part
        )
        if cross_attention_dim is None:
            self.to_k_oft = TTMLoRALinearLayer(
                hidden_size, hidden_size,
                in_tensor_shape=hidden_tensor_shape, out_tensor_shape=hidden_tensor_shape,
                rank=rank, transform_rank=transform_rank,
                alpha=alpha, use_rslora=use_rslora, addition_part=addition_part
            )
            self.to_v_oft = TTMLoRALinearLayer(
                hidden_size, hidden_size,
                in_tensor_shape=hidden_tensor_shape, out_tensor_shape=hidden_tensor_shape,
                rank=rank, transform_rank=transform_rank,
                alpha=alpha, use_rslora=use_rslora, addition_part=addition_part
            )
        else:
            self.to_k_oft = TTMLoRALinearLayer(
                cross_attention_dim, hidden_size,
                in_tensor_shape=cross_tensor_shape, out_tensor_shape=hidden_tensor_shape,
                rank=rank, transform_rank=transform_rank,
                alpha=alpha, use_rslora=use_rslora, addition_part=addition_part
            )
            self.to_v_oft = TTMLoRALinearLayer(
                cross_attention_dim, hidden_size,
                in_tensor_shape=cross_tensor_shape, out_tensor_shape=hidden_tensor_shape,
                rank=rank, transform_rank=transform_rank,
                alpha=alpha, use_rslora=use_rslora, addition_part=addition_part
            )
        self.to_out_oft = TTMLoRALinearLayer(
            hidden_size, hidden_size,
            in_tensor_shape=hidden_tensor_shape, out_tensor_shape=hidden_tensor_shape,
            rank=rank, transform_rank=transform_rank,
            alpha=alpha, use_rslora=use_rslora, addition_part=addition_part
        )

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        
        query = self.to_q_oft(attn.to_q, hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        key = self.to_k_oft(attn.to_k, encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        value = self.to_v_oft(attn.to_v, encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        # hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = self.to_out_oft(attn.to_out[0], hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
