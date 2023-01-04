# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import collections
from typing import Optional, List, Dict, Any

import paddle
import paddle.nn as nn
import paddle.incubate as incubate
from paddle.nn import LayerNorm, Layer
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
    get_rng_state_tracker,
)
from paddle.nn.layer.transformer import _convert_param_attr_to_list
#from paddlenlp.transformers.gpt.modeling import MultiHeadAttention, TransformerDecoderLayer
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    # rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits

class TransformerDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.
    It contains multiheadattention and some linear layers.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="gelu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
        num_partitions=1,
        fuse=False,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()

        self.fuse = fuse
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        if self.fuse:
            hcg = fleet.get_hybrid_communicate_group()
            mp_nranks = hcg.get_model_parallel_world_size()
            mp_group = hcg.get_model_parallel_group()
            ring_id = mp_group.id if mp_nranks > 1 else -1
            self.self_attn = incubate.nn.FusedMultiHeadAttention(
                d_model,
                nhead,
                dropout_rate=dropout,
                attn_dropout_rate=attn_dropout,
                normalize_before=normalize_before,
                qkv_weight_attr=weight_attrs[0],
                qkv_bias_attr=bias_attrs[0],
                linear_weight_attr=weight_attrs[0],
                linear_bias_attr=bias_attrs[0],
                epsilon=1e-5,
                nranks=mp_nranks,
                ring_id=ring_id,
            )
            self.ffn = incubate.nn.FusedFeedForward(
                d_model,
                dim_feedforward,
                dropout_rate=act_dropout,
                epsilon=1e-5,
                activation=activation,
                normalize_before=normalize_before,
                act_dropout_rate=0.0,
                linear1_weight_attr=weight_attrs[2],
                linear1_bias_attr=bias_attrs[2],
                linear2_weight_attr=weight_attrs[2],
                linear2_bias_attr=bias_attrs[2],
                nranks=mp_nranks,
                ring_id=ring_id,
            )
        else:
            self.self_attn = MultiHeadAttention(
                d_model,
                nhead,
                dropout=attn_dropout,
                weight_attr=weight_attrs[0],
                bias_attr=bias_attrs[0],
                num_partitions=num_partitions,
            )

            self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
                d_model, dim_feedforward, weight_attr=weight_attrs[2], gather_output=False, has_bias=True
            )

            self.linear2 = fleet.meta_parallel.RowParallelLinear(
                dim_feedforward, d_model, weight_attr=weight_attrs[2], input_is_parallel=True, has_bias=True
            )

            self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
            self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
            self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
            self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
            self.activation = getattr(F, activation)

    def forward(self, tgt, tgt_mask=None, memory=None,  use_cache=False, cache=None):
        if tgt_mask is None:
            causal_mask = paddle.tensor.triu(
            paddle.ones((paddle.shape(tgt)[-1], paddle.shape(tgt)[-1])) * -1e4, diagonal=1
        )
            tgt_mask = causal_mask

        if self.fuse:
            if use_cache:
                attn_output, cache_kv_out = self.self_attn(tgt, attn_mask=tgt_mask, cache=cache.kv)
            else:
                attn_output = self.self_attn(tgt, attn_mask=tgt_mask)

            enc_out = self.ffn(attn_output)
            return (enc_out, cache_kv_out) if use_cache else enc_out

        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)

        tgt = residual + self.dropout1(tgt)

        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)

        tgt = self.dropout2(self.linear2(F.relu(self.linear1(tgt))))

        tgt = residual + tgt

        if not self.normalize_before:
            tgt = self.norm2(tgt)
        #print(tgt[0,0,:100])
        #import pdb; pdb.set_trace()

        return tgt if use_cache is False else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return incremental_cache

class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
        fuse=False,
        num_partitions=1,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse = fuse

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self.num_heads % num_partitions == 0
        self.num_heads = self.num_heads // num_partitions

        if self.fuse:
            assert self.kdim == embed_dim, "embed_dim should be equal to kdim"
            assert self.vdim == embed_dim, "embed_dim should be equal to vidm"

            self.qkv_proj = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim, 3 * embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )
        else:
            self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim, embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )

            self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.kdim, embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )

            self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.vdim, embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )

        self.out_proj = fleet.meta_parallel.RowParallelLinear(
            embed_dim, embed_dim, weight_attr=weight_attr, has_bias=True, input_is_parallel=True
        )

    def _fuse_prepare_qkv(self, query):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, self.num_heads, 3 * self.head_dim])
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)
        return q, k, v

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.
        """
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v) if use_cache is False else (q, k, v, cache)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.
        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.
        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key, shape=[-1, self.num_heads, 0, self.head_dim], dtype=key.dtype, value=0
            )
            v = layers.fill_constant_batch_size_like(
                input=key, shape=[-1, self.num_heads, 0, self.head_dim], dtype=key.dtype, value=0
            )
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key, value, attn_mask=None, use_cache=False, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if use_cache is False:
            if self.fuse:
                q, k, v = self._fuse_prepare_qkv(query)
            else:
                q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache, cache)
        # scale dot product attention
        product = paddle.matmul(x=q * (self.head_dim**-0.5), y=k, transpose_y=True)

        # if attn_mask is not None:
        # product = product + attn_mask
        # weights = F.softmax(product)

        weights = incubate.softmax_mask_fuse_upper_triangle(product)

        if self.dropout:
            with get_rng_state_tracker().rng_state("local_seed"):
                weights = F.dropout(weights, self.dropout, training=self.training, mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


__all__ = [
    "OPTModel",
    "OPTPretrainedModel",
    "OPTForCausalLM",
]

class TransformerDecoder(Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(
        self,
        decoder_layers: List[Layer],
        num_layers: int,
        hidden_size: int,
        word_embed_proj_dim: int,
        norm: Optional[Layer] = None,
        normalize_before: bool = False,
    ):
        super(TransformerDecoder, self).__init__()

        if word_embed_proj_dim != hidden_size:
            self.project_out = nn.Linear(hidden_size, word_embed_proj_dim, bias_attr=False)
        else:
            self.project_out = None

        self.num_layers = num_layers
        self.layers = decoder_layers

        if normalize_before:
            self.final_layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.final_layer_norm = None

        self.checkpoints = []

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None, use_cache: bool = False, cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []
        self.checkpoints = []

        for i, mod in enumerate(self.layers):
            if cache is None:
                if use_cache:
                    output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)
                    new_caches.append(new_cache)
                else:
                    output = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)

            else:
                output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache[i])
                new_caches.append(new_cache)
            self.checkpoints.append(output.name)

        if self.final_layer_norm:
            output = self.final_layer_norm(output)

        if self.project_out:
            output = self.project_out(output)

        return output if use_cache is False else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
        """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class OPTLearnedPositionEmbedding(nn.Embedding):
    """this module learns postional embeddings up to a fixed maximum size"""

    def __init__(self, num_embeddings: int, embedding_dim: int, initializer_range: float):
        """OPT is set up so taht if padding_idx is specified then offset the embedding ids by 2
        and adjust num_embeddings appropriately. Other models don't have this hack

        Args:
            num_embeddings (int): the number of embedding size
            embedding_dim (int): the dim of embedding
        """
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, position_ids, past_key_values_length: int = 0):
        """get the position embedding with attention mask

        Args:
            position_ids: (paddle.Tensor): the tensor of position ids
            past_key_values_length (int, optional): the past key value which will . Defaults to 0.

        Returns:
            paddle.Tensor: the position embedding
        """
        # cut positions if `past_key_values_length` is > 0
        position_ids = position_ids[:, past_key_values_length:]
        return super().forward(position_ids + self.offset)


class OPTEmbeddings(Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        word_embed_proj_dim: int = 768,
        padding_idx: int = 1,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: Optional[int] = None,
        initializer_range=0.02,
    ):
        super(OPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size,
            word_embed_proj_dim,
            # padding_idx=padding_idx,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        if word_embed_proj_dim != hidden_size:
            self.project_in = nn.Linear(word_embed_proj_dim, hidden_size, bias_attr=False)
        else:
            self.project_in = None

        self.position_embeddings = OPTLearnedPositionEmbedding(
            num_embeddings=max_position_embeddings, embedding_dim=hidden_size, initializer_range=initializer_range
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        input_embeddings = self.word_embeddings(input_ids)

        if self.project_in:
            input_embeddings = self.project_in(input_embeddings)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class OPTPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained OPT models. It provides OPT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = {}
    pretrained_resource_files_map = {"model_state": {}}
    base_model_prefix = "opt"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.opt.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )

class EmbeddingPipe(OPTEmbeddings):
    """Extends GPTEmbeddings to forward attention_mask through the pipeline."""

    @property
    def embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, tensors):
        input_ids, position_ids = tensors
        embeddings = super().forward(input_ids=input_ids, position_ids=position_ids)
        return embeddings

@register_base_model
class OPTModel(OPTPretrainedModel,PipelineLayer):
    r"""
    The bare OPT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `OPTModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `OPTModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer and decoder layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer decoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the decoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"relu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and decoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all decoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`. Defaults to `16`.

            .. note::
                Please NOT using `type_vocab_size`, for it will be obsolete in the future..

        initializer_range (float, optional):
            The standard deviation of the normal initializer. Default to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`OPTPretrainedModel._init_weights()` for how weights are initialized in `OPTModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
             to `0`.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        word_embed_proj_dim: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        eos_token_id: int = 7,
        bos_token_id: int = 0,
        eol_token_id: int = 3,
        normalize_before: bool = True,
        p_degree = 1,
        **kwargs
    ):
        OPTPretrainedModel.__init__(self)
        


        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.descs = []
        self.descs.append(
            SharedLayerDesc(
                "embeddings",
                EmbeddingPipe,
                shared_weight_attr="embedding_weight",
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                word_embed_proj_dim=word_embed_proj_dim,
                padding_idx=pad_token_id,
                hidden_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=type_vocab_size,
                initializer_range=initializer_range,
            )
        )

        for _ in range(num_hidden_layers):
            self.descs.append(
                LayerDesc(
                    TransformerDecoderLayer,
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    normalize_before = True,
                    dim_feedforward=intermediate_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
                    bias_attr=None,
                )
            )
        # opt-345m, layer, normalize_before=False, no layrnorm
        #self.descs.append(LayerDesc(nn.Linear, hidden_size, word_embed_proj_dim, bias_attr=False))
        #> opt-1.3b layrnorm normalize_beforre=True
        self.descs.append(LayerDesc(nn.LayerNorm, normalized_shape=hidden_size))

        
        #'''
        def _logits_helper(embedding, output):
            return parallel_matmul(output, embedding.embedding_weight, True)

        self.descs.append(
            SharedLayerDesc(
                "embeddings",
                EmbeddingPipe,
                forward_func=_logits_helper,
                shared_weight_attr="embedding_weight",
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                word_embed_proj_dim=word_embed_proj_dim,
                padding_idx=pad_token_id,
                hidden_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=type_vocab_size,
                initializer_range=initializer_range,
            )
        )
        #'''
        
        hcg = fleet.get_hybrid_communicate_group()

        PipelineLayer.__init__(self,
            layers=self.descs,
            topology=hcg.topology(),
            seg_method="layer:TransformerDecoderLayer",
            )



class OPTLMHead(Layer):
    def __init__(self, hidden_size: int, vocab_size: int, embedding_weights=None):
        super(OPTLMHead, self).__init__()
        self.decoder_weight = (
            self.create_parameter(shape=[vocab_size, hidden_size], dtype=paddle.get_default_dtype(), is_bias=True)
            if embedding_weights is None
            else embedding_weights
        )

    def forward(self, hidden_states):
        logits = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True)
        return logits


class OPTForCausalLM(OPTPretrainedModel,PipelineLayer):
    """
    The OPT Model with a `language modeling` head on top.

    Args:
        opt (:class:`OPTModel`):
            An instance of :class:`OPTModel`.

    """

    def __init__(self, opt: OPTModel):
        super(OPTForCausalLM, self).__init__()
        self.lm_head = OPTLMHead(
            hidden_size=self.opt.config["hidden_size"],
            vocab_size=self.opt.config["vocab_size"],
            embedding_weights=self.opt.embeddings.word_embeddings.weight,
        )

    def forward(self, input_ids, position_ids=None, attention_mask=None, use_cache=False, cache=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`OPTModel`.
            position_ids (Tensor, optional):
                See :class:`OPTModel`.
            attention_mask (Tensor, optional):
                See :class:`OPTModel`.
            use_cache (bool, optional):
                See :class:`OPTModel`.
            cache (Tensor, optional):
                See :class:`OPTModel`.

        Returns:
            Tensor or tuple: Returns tensor `logits` or tuple `(logits, cached_kvs)`. If `use_cache` is True,
            tuple (`logits, cached_kvs`) will be returned. Otherwise, tensor `logits` will be returned.
            `logits` is the output of the opt model.
            `cache_kvs` is the cache output of opt model if `use_cache` is True.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import OPTForCausalLM, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('facebook/opt-125m')
                model = OPTForCausalLM.from_pretrained('facebook/opt-125m')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output_ids, score = model.generate(input_ids=inputs['input_ids'])
                print(tokenizer.batch_decode(output_ids[0]))
        """

        outputs = self.opt(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )

        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = self.lm_head(encoder_outputs)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def prepare_faster_entry(self, kwargs: Dict[str, Any]):
        # import FasterOPT at here to avoid cycling import
        from paddlenlp.ops import FasterOPT

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decode_strategy = kwargs.get("decode_strategy")
        # decoding_lib can be passed into FasterOPT
        decoding_lib = kwargs.get("decoding_lib", None)

        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the faster version of OPT")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.opt.config["hidden_size"] // self.opt.config["num_attention_heads"]
        if size_per_head not in [32, 64, 80, 96, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the faster version of OPT" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the faster version")
        if kwargs["min_length"] != 0:
            # not support for min_length yet in the faster version
            raise AttributeError("'min_length != 0' is not supported yet in the faster version")
        self._faster_entry = FasterOPT(self, use_fp16_decoding=use_fp16_decoding, decoding_lib=decoding_lib).forward
        return self._faster_entry

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.common_ops_import.convert_dtype(attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                position_ids += 2
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e
