# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import copy
import inspect
import io
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import paddle
import paddle.nn as nn
import six
from paddlenlp.utils.log import logger

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def load(model, weight_path, base_model_prefix='opt', max_layer = 24):
    state_dict = paddle.load(weight_path)

    model_to_load = model
    state_to_load = state_dict
    unexpected_keys = []
    missing_keys = []
    
    if not hasattr(model, base_model_prefix) and any(
            s.startswith(base_model_prefix) for s in state_dict.keys()
        ):
            # base model
            state_to_load = {}
            start_prefix = base_model_prefix + "."
            for k, v in state_dict.items():
                if k.startswith(base_model_prefix):
                    state_to_load[k[len(start_prefix) :]] = v
                else:
                    unexpected_keys.append(k)
    if hasattr(model, base_model_prefix) and not any(
        s.startswith(base_model_prefix) for s in state_dict.keys()
    ):
        # derived model (base model with heads)
        model_to_load = getattr(model, base_model_prefix)
        for k in model.state_dict().keys():
            if not k.startswith(base_model_prefix):
                missing_keys.append(k)
    if len(missing_keys) > 0:
        logger.info(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        logger.info(
            "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
        )

    dtype_prefix_len = len("paddle.")
    print(model_to_load.state_dict().keys())
    print(state_to_load.keys())
    for k, v in model_to_load.state_dict().items():
        layer_name = copy.copy(k)
        k = k.replace('_layers.','')
        k = k.replace('sharedembeddings','embeddings')
        if not isinstance(v, np.ndarray):
            dtype = str(v.dtype)[dtype_prefix_len:]
        if has_numbers(k):
            idx = -1
            layer_index  = int(re.search(r'\d+', k).group())
            k = k.replace(str(layer_index),str(layer_index-1),1)
            k = 'decoder.layers.' +  k
            if layer_index == (max_layer+1):
                k = 'decoder.project_out.weight'
                if k not in state_to_load:
                    k = 'decoder.final_layer_norm' + k[len(str(layer_index)):]
                    if k not in state_to_load:
                        import pdb; pdb.set_trace()

        if k in state_to_load:
            if paddle.in_dynamic_mode():
                if isinstance(state_to_load[k], np.ndarray):
                    state_to_load[k] = state_to_load[k].astype(dtype)
                else:
                    state_to_load[k] = paddle.cast(state_to_load[k], dtype)
            else:
                state_to_load[k] = np.array(state_to_load[k])
                state_to_load[k] = state_to_load[k].astype(dtype)
            state_to_load[layer_name] = state_to_load[k]
            '''
            if 'embedding' in layer_name:
                print(v)
                print(layer_name)
                print(model_to_load.state_dict()[layer_name])
            '''
        else:
            print(k)
            #abc
           
    import paddlenlp.ops.faster_transformer.transformer.decoding as ft_decoding
    state_to_load = ft_decoding.get_ft_para_conf().fit_partial_model(model_to_load, state_to_load)
    model_to_load.set_state_dict(state_to_load)
    return model_to_load
