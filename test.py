import random
import sys
import os
import importlib.util
import time

import pandas as pd
import numpy as np
#from main import trajectory
from data_preprocess.clean_data import read_data, data_extract, interpolation_data_by_samples,save_data


if __name__ == '__main__':

    # np.random.seed(int(time.time()))
    # y_t = np.random.uniform(0.0,3.0, size=(20,2))
    # y_p = np.random.uniform(0.0,3.0, size=(20,2))
    # trajectory(y_p, y_t)
    import keras
    import tensorflow as tf

    print(keras.__version__)
    print(tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    # tof_df = pd.read_csv("./clean_data/preprocessed-TOFEXP1.csv")  # 包含 'Timestamp' 列
    # ultra_df = pd.read_csv("./raw_data/exp1_10min_ultrasound.csv")  # 包含 'Timestamp', 'X', 'Y' 列

    # # 确保时间戳升序排列
    # tof_df = tof_df.sort_values(".HostTimestamp")
    # ultra_df = ultra_df.sort_values("Timestamp")

    # 使用 merge_asof 进行最近邻对齐（左对齐）
    # merged_df = pd.merge_asof(tof_df, ultra_df, on="Timestamp", direction="nearest")
    # merged_df.to_csv('./raw_data/www.csv')


    # tof_timestamps = np.array(tof_df['.HostTimestamp'])
    # ultra_timestamps = np.array(ultra_df['Timestamp'])
    #
    # # 查找每个 TOF 时间戳在 ultrasound 中的最近索引
    # idx = np.abs(ultra_timestamps[:, None] - tof_timestamps).argmin(axis=0)
    #
    # # 得到与 tof_df 对应的 (X, Y)
    # ultra_X = ultra_df.iloc[idx]['X'].values
    # ultra_Y = ultra_df.iloc[idx]['Y'].values
    #
    # # 合并结果
    # tof_df['X'] = ultra_X
    # tof_df['Y'] = ultra_Y
    # tof_df.to_csv('./raw_data/www.csv')



