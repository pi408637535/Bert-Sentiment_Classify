# -*- coding: utf-8 -*-
# @Time    : 2020/9/3 14:39
# @Author  : piguanghua
# @FileName: run.py
# @Software: PyCharm


import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import get_time_dif
from data_utils import build_dataset
import pandas as pd
from data_utils import convert_examples_to_features
import os

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--max_seq_length', type=int, default=365, help='maximum total input sequence length')
parser.add_argument('--split_num', type=int, default=3, help='split_num')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




args = parser.parse_args()





if __name__ == '__main__':
    dataset = 'text_emotion'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(1)

    start_time = time.time()
    print("Loading data...")



    train_iter, dev_iter = build_dataset(config, args)
    #train_iter = build_iterator(train_data, config)
    #dev_iter = build_iterator(dev_data, config)
    #test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, None)