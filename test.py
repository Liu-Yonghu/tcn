import random
import sys
import os
import importlib.util
import time

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.datasets import load_files

def create_tf_dataset(data_array, output_array, input_sequence_length, output_sequence_length, batch_size=1):
    inputs = timeseries_dataset_from_array(
        data_array[:-2, :], # (8,4,1)
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = math.floor(input_sequence_length / 2) + 1  # 3
    targets = timeseries_dataset_from_array(
        output_array[target_offset:-target_offset, :],  # output_array[3:-3] => (4,2)
        None,
        sequence_length=output_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    return dataset

def create_dataset_window(X_all, Y_all, window_size=5):
    X_list = []
    Y_list = []

    for i in range(len(X_all) - window_size + 1):
        X_seq = X_all[i: i + window_size, :]        # shape (20, 64)
        #Y_target = Y_all[i + window_size-1]
        Y_target = Y_all[i + np.floor(window_size / 2).astype(int) + 1, :]     # from  middle smaples

        X_list.append(X_seq)
        Y_list.append(Y_target)

    X = np.stack(X_list)  # (N, 15, 64)
    Y = np.stack(Y_list)  # (N, 2)
    return X, Y


if __name__ == '__main__':
    print(keras.__version__)
    print(tf.__version__)
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.utils import timeseries_dataset_from_array
    import math

    # 假设输入数据 10 步，每步 4 特征
    data_array = np.arange(400).reshape(100, 4)  # (10,4)
    # 假设输出数据 10 步，每步 (a,b)
    output_array = np.arange(200).reshape(100, 2)  # (10,2)
    print(data_array)

    input_sequence_length = 5
    output_sequence_length = 1
    ds = create_tf_dataset(data_array, output_array, 5, 1, batch_size=32)
    for x, y in ds:
        print("X.shape:", x.shape, "Y.shape:", y.shape)
        print("X:", x.numpy(), "Y:", y.numpy())
    print("===========================================")
    X, Y = create_dataset_window(data_array, output_array, 5)
    print("X.shape:", X.shape, "Y.shape:", Y.shape)
    for i in range(len(X)):
        print("X[{}]:".format(i), X[i][:, ], "Y:", Y[i])



