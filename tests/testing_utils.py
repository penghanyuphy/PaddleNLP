# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import unittest
import paddle
import os
import inspect
from distutils.util import strtobool
from collections.abc import Mapping
import gc

__all__ = ["get_vocab_list", "stable_softmax", "cross_entropy"]


class PaddleNLPModelTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()


def get_vocab_list(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = [vocab.rstrip("\n").split("\t")[0] for vocab in f.readlines()]
        return vocab_list


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1 :]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def softmax_with_cross_entropy(logits, label, soft_label=False, axis=-1, ignore_index=-1):
    softmax = np.apply_along_axis(stable_softmax, -1, logits)
    return cross_entropy(softmax, label, soft_label, axis, ignore_index)


def assert_raises(Error=AssertionError):
    def assert_raises_error(func):
        def wrapper(self, *args, **kwargs):
            with self.assertRaises(Error):
                func(self, *args, **kwargs)

        return wrapper

    return assert_raises_error


def create_test_data(file=__file__):
    dir_path = os.path.dirname(os.path.realpath(file))
    test_data_file = os.path.join(dir_path, "dict.txt")
    with open(test_data_file, "w") as f:
        vocab_list = [
            "[UNK]",
            "AT&T",
            "B???",
            "c#",
            "C#",
            "c++",
            "C++",
            "T???",
            "A???",
            "A???",
            "A???",
            "A???",
            "AA???",
            "AB???",
            "B???",
            "B???",
            "B???",
            "B???",
            "BB???",
            "BP???",
            "C???",
            "C???",
            "C??????",
            "CD???",
            "CD???",
            "CALL???",
            "D???",
            "D???",
            "D???",
            "E???",
            "E???",
            "E???",
            "E???",
            "F???",
            "F???",
            "G???",
            "H???",
            "H???",
            "I???",
            "IC???",
            "IP???",
            "IP??????",
            "IP??????",
            "K???",
            "K?????????",
            "N???",
            "O???",
            "PC???",
            "PH???",
            "SIM???",
            "U???",
            "VISA???",
            "Z???",
            "Q???",
            "QQ???",
            "RSS??????",
            "T???",
            "X???",
            "X??????",
            "X??????",
            "????????",
            "T??????",
            "T??????",
            "T???",
            "4S???",
            "4s???",
            "??????style",
            "??????Style",
            "1??????",
            "???S",
            "???S",
            "???Q",
            "???",
            "??????",
            "?????????",
            "?????????",
            "?????????",
            "????????????",
            "?????????",
            "????????????",
            "?????????",
            "????????????",
            "??????",
            "????????????",
            "?????????",
            "????????????",
            "??????",
            "????????????",
            "??????",
            "????????????",
            "???????????????????????????",
            "???????????????????????????",
            "??????????????????",
            "???????????????????????????",
            "????????????",
            "???????????????",
            "???????????????",
            "??????????????????",
            "?????????",
        ]
        for vocab in vocab_list:
            f.write("{}\n".format(vocab))
    return test_data_file


def get_bool_from_env(key, default_value=False):
    if key not in os.environ:
        return default_value
    value = os.getenv(key)
    try:
        value = strtobool(value)
    except ValueError:
        raise ValueError(f"If set, {key} must be yes, no, true, false, 0 or 1 (case insensitive).")
    return value


_run_slow_test = get_bool_from_env("RUN_SLOW_TEST")


def slow(test):
    """
    Mark a test which spends too much time.
    Slow tests are skipped by default. Excute the command `export RUN_SLOW_TEST=True` to run them.
    """
    if not _run_slow_test:
        return unittest.skip("test spends too much time")(test)
    else:
        return test


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


def nested_simplify(obj, decimals=3):
    """
    Simplifies an object by rounding float numbers, and downcasting tensors/numpy arrays to get simple equality test
    within tests.
    """
    import numpy as np

    if isinstance(obj, list):
        return [nested_simplify(item, decimals) for item in obj]
    elif isinstance(obj, np.ndarray):
        return nested_simplify(obj.tolist())
    elif isinstance(obj, Mapping):
        return {nested_simplify(k, decimals): nested_simplify(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (str, int, np.int64)):
        return obj
    elif obj is None:
        return obj
    elif isinstance(obj, paddle.Tensor):
        return nested_simplify(obj.numpy().tolist(), decimals)
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, (np.int32, np.float32)):
        return nested_simplify(obj.item(), decimals)
    else:
        raise Exception(f"Not supported: {type(obj)}")
