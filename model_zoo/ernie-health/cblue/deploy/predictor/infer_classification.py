# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse
import psutil

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset

from predictor import CLSPredictor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path", default="ernie-health-chinese", type=str, help="The directory or name of model.")
parser.add_argument("--dataset", default="KUAKE-QIC", type=str, help="Dataset for text classfication.")
parser.add_argument("--data_file", default=None, type=str, help="The data to predict with one sample per line.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--use_fp16", action='store_true', help="Whether to use fp16 inference, only takes effect when deploying on gpu.")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads", default=psutil.cpu_count(logical=False), type=int, help="num_threads for cpu.")
parser.add_argument("--device", choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--device_id", default=0, help="Select which gpu device to train model.")
args = parser.parse_args()
# yapf: enable

LABEL_LIST = {
    "kuake-qic": ["????????????", "????????????", "????????????", "????????????", "????????????", "????????????", "????????????", "????????????", "????????????", "????????????", "??????"],
    "kuake-qtr": ["???????????????", "????????????????????????????????????", "????????????", "????????????"],
    "kuake-qqr": ["B???A??????????????????B??????????????????A??? ??????A???B?????????????????????", "B???A??????????????????B??????????????????A???", "??????A???B??????????????????????????????"],
    "chip-ctc": [
        "????????????",
        "????????????",
        "??????",
        "????????????",
        "????????????",
        "??????",
        "??????",
        "??????",
        "?????????",
        "????????????",
        "???????????????",
        "??????",
        "??????",
        "??????",
        "????????????",
        "??????",
        "????????????",
        "????????????",
        "??????????????????",
        "????????????",
        "??????",
        "??????",
        "??????",
        "????????????",
        "???????????????",
        "????????????",
        "????????????",
        "????????????????????????",
        "????????????",
        "????????????",
        "??????",
        "????????????",
        "??????????????????",
        "??????",
        "????????????",
        "????????????",
        "???????????????",
        "????????????",
        "?????????",
        "??????(???????????????",
        " ????????????",
        "??????????????????",
        "??????(????????????)",
        "???????????????",
    ],
    "chip-sts": ["????????????", "????????????"],
    "chip-cdn-2c": ["???", "???"],
}

TEXT = {
    "kuake-qic": ["???????????????????????????????????????", "??????????????????????????????????????????"],
    "kuake-qtr": [["?????????????????????????????????", "????????????????????????????????????????????????"], ["????????????????????????", "????????????????????????????????????"]],
    "kuake-qqr": [["??????????????????", "??????????????????"], ["???????????????????????????", "???????????????????????????"]],
    "chip-ctc": ["(1)??????????????????????????????????????????????????????????????????????????????", "??????????????????????????????????????????"],
    "chip-sts": [["?????????????????????????????????????????????", "????????????????????????????????????"], ["H?????????????????????", "WHO?????????????????????????????????????????????"]],
    "chip-cdn-2c": [["1?????????????????????????????????", " 1??????????????????IV???"], ["?????????????????????", "????????????"]],
}

METRIC = {
    "kuake-qic": "acc",
    "kuake-qtr": "acc",
    "kuake-qqr": "acc",
    "chip-ctc": "macro",
    "chip-sts": "macro",
    "chip-cdn-2c": "macro",
}

if __name__ == "__main__":
    for arg_name, arg_value in vars(args).items():
        logger.info("{:20}: {}".format(arg_name, arg_value))

    args.dataset = args.dataset.lower()
    label_list = LABEL_LIST[args.dataset]
    if args.data_file is not None:
        with open(args.data_file, "r") as fp:
            input_data = [x.strip().split("\t") for x in fp.readlines()]
            input_data = [x[0] if len(x) == 1 else x for x in input_data]
    else:
        input_data = TEXT[args.dataset]

    predictor = CLSPredictor(args, label_list)
    predictor.predict(input_data)
