# -*- coding: utf-8 -*-
# @Time    : 2020/9/3 15:55
# @Author  : piguanghua
# @FileName: raw_text_process.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import seaborn as sns



if __name__ == '__main__':
    source_path = "/Users/piguanghua/Downloads/text_emotion/train.csv"
    train_df=pd.read_csv(source_path)

    #show_columns(df) 'ID', 'txt', 'Label'

    print(train_df["Label"].unique())
    print(train_df.shape)

    #remove label is empty
    train_df['Label'] = train_df['Label'].fillna(-1)
    train_df = train_df[train_df['Label'] != -1]
    train_df['txt'].fillna('none')
    train_df['txt'] = train_df['txt'].fillna('none')

    train_index = train_df.shape[0]
    train_txt = "/Users/piguanghua/Downloads/text_emotion/train.txt"
    dev_txt = "/Users/piguanghua/Downloads/text_emotion/dev.txt"

    train_index = int(train_index * 0.9)
    dev_df = train_df.iloc[train_index:]
    train_df = train_df.iloc[:train_index]


    with open(train_txt, "w") as f_train:
        for index,row in train_df.iterrows():
            line = row["txt"] + "\t" + str(row["Label"]) + "\n"
            f_train.write(line)

    with open(dev_txt, "w") as f_dev:
        for index, row in dev_df.iterrows():
            line = row["txt"] + "\t" + str(row["Label"]) + "\n"
            f_dev.write(line)


