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

def show_columns(df):
    print(train_df.columns)

if __name__ == '__main__':
    #source_path = "/Users/piguanghua/Downloads/text_emotion/train.csv"
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

    text_len = [len(i) for i in train_df['txt']]
    y = sorted(text_len)
    print("max ", y[-1])
    print("min ", y[0])
    x = np.linspace(0, y[-1], num=len(y))

    plt.figure()
    plt.plot(list(x), y)
    plt.show()

    data = [
            (train_df['Label'][train_df['Label'] == 0]).shape[0],
            (train_df['Label'][train_df['Label'] == 1]).shape[0]
            ]

    plt.hist([0,1], bins=data, facecolor="blue", edgecolor="black")

    sns.barplot(x=[0,1], y =data )

    plt.show()


