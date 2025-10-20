import random
import sys
import importlib.util
import time
import math
import os
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import models, Input
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.utils import timeseries_dataset_from_array
from tcn_old import TCN
from keras import layers, models

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

def model_TCN(hidden, num_filters, k_size, dense):
    x = layers.Input(shape=(20, 64))
    tcn_out = TCN(nb_filters=num_filters, kernel_size=k_size, nb_stacks=1, dilations=[2 ** i for i in range(hidden)],
                  padding='same', use_skip_connections='True', dropout_rate=0.01, return_sequences=False,
                  activation='relu', kernel_initializer='glorot_uniform', use_layer_norm=True, name='tcn')(x)
    flatten_out = layers.Flatten()(tcn_out)
    dense_out1 = layers.Dense(dense)(flatten_out)
    print("shape_out_tcn:", tcn_out.shape)
    dense_out = layers.Dense(2)(dense_out1)
    print("dense_out:", dense_out.shape)
    model = models.Model(x, dense_out)
    # model.load_weights(path_tcn)
    return model

def rebuild_model(trial_dir, input_shape=(20, 64), ckpt_exec=8):
        # 1. 读取 trial.json
        trial_json = os.path.join(trial_dir, "trial.json")
        with open(trial_json, "r") as f:
            trial_cfg = json.load(f)

        params = trial_cfg["hyperparameters"]["values"]
        print("Loaded hyperparameters:", params)

        # 2. 构建模型 (和 build(self,hp) 一致)
        x = Input(shape=input_shape)
        tcn_model = model_TCN(
            hidden=params["hidden"],
            num_filters=params["nb_filters"],
            k_size=params["k_size"],
            dense=params["dense"]
        )
        tcn_output = tcn_model(x)
        full_model = models.Model(inputs=x, outputs=tcn_output)

        # 3. 加载权重Tckpt_exec2.weights.h5
        ckpt_file = os.path.join(trial_dir, f"ckpt_exec{ckpt_exec}.weights.h5")
        print(ckpt_file)
        if os.path.exists(ckpt_file):
            full_model.load_weights(ckpt_file)
            print(f"Loaded weights from {ckpt_file}")
        else:
            print(f"Checkpoint not found: {ckpt_file}")

        return full_model

    # ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # User: yonghu
    # Password: cacNuphUb4
    # ** ** ** ** ** ** ** ** ** ** ** ** ** **

if __name__ == '__main__':
    # print(keras.__version__)
    # print(tf.__version__)
    #
    #
    #
    # data_array = np.arange(400).reshape(100, 4)  # (10,4)
    #
    # output_array = np.arange(200).reshape(100, 2)  # (10,2)
    # print(data_array)
    #
    # input_sequence_length = 5
    # output_sequence_length = 1
    # ds = create_tf_dataset(data_array, output_array, 5, 1, batch_size=32)
    # for x, y in ds:
    #     print("X.shape:", x.shape, "Y.shape:", y.shape)
    #     print("X:", x.numpy(), "Y:", y.numpy())
    # print("===========================================")
    # X, Y = create_dataset_window(data_array, output_array, 5)
    # print("X.shape:", X.shape, "Y.shape:", Y.shape)
    # for i in range(len(X)):
    #     print("X[{}]:".format(i), X[i][:, ], "Y:", Y[i])


    # try to rebuild model
    # trial_dir = "./results/std_TOFEXP1/6/trial_12"
    # model = rebuild_model(trial_dir, input_shape=(20, 64), ckpt_exec=1)
    #
    # # 打印结构
    # model.summary()
    #
    # # 示例推理
    # dummy_input = np.random.rand(1, 20, 64).astype(np.float32)
    # pred = model.predict(dummy_input)
    # print("Dummy prediction:", pred)
    # df = pd.read_csv("./exp_data/norm_TOFEXP1.csv")
    # df = df.round(6)
    # df.to_csv('./exp_data/norm_TOFEXP1_6.csv', index=False, header=False)
    # print(df.head())
    #
    # num = input(" please input an int number")

    df = pd.read_csv("./exp_data/norm_mmWaveEXP2.csv").head(2155)
    df.to_csv("./exp_data/norm_mmWaveEXP2_1.csv", index=False, header=False)











