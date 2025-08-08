import keras.optimizers
import os
import utils
from tcn_simple import TCN_model
import pandas as pd
import tcn_model
from math import inf
import numpy as np
import tcn_model as tm
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import argparse
from utils import rmse


def create_dataset_window(X_all, Y_all, window_size=20):
    X_list = []
    Y_list = []

    for i in range(len(X_all) - window_size + 1):
        X_seq = X_all[i: i + window_size]        # shape (15, 64)
        Y_target = Y_all[i + window_size - 1]     # 第15个样本的标签

        X_list.append(X_seq)
        Y_list.append(Y_target)

    X = np.stack(X_list)  # (N, 15, 64)
    Y = np.stack(Y_list)  # (N, 2)
    return X, Y

def trajectory_curve(labels, GT_value, info=None):

    X_T = GT_value[:, :1]
    Y_T = GT_value[:, 1:]
    X_P = labels[:, :1]
    Y_P= labels[:, 1:]
    # 创建图形
    plt.figure(figsize=(6, 6))
    plt.plot(X_P, Y_P, marker='x', linestyle='-', color='blue', label='Prediction')
    plt.plot(X_T, Y_T, marker='o', linestyle='--', color='red', label='Ground True')
    plt.xlim(0, 3)
    plt.ylim(0, 3)

    # 设置标题和标签
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Ground Truth vs Prediction Trajectory')
    plt.grid(True)
    plt.text(0.02, 0.12, "Model: TCN\nMAE: 0.14\nExp: 1",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.legend()

    # 保存图像
    save_path = './results/images'
    os.makedirs(save_path,exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{info["exp"]}-{info["slicing"]}-{info["model"]}-trajectory_comparison.png'), dpi=600)


def loss_curve(history, info=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = './results/images'
    os.makedirs(save_path,exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{info["exp"]}-{info["slicing"]}-{info["model"]}-loss_comparison.png'), dpi=600)

def slice_dataset(num_data, training_part, validing_part):
    # slide the data into three part
    # n = len(df)
    train_end = int(training_part * num_data)  # 3/(3+1+1) = 0.6
    val_end = int((training_part+validing_part) * num_data)  # 0.6 + 0.2 = 0.8

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    # 保存为新的 CSV 文件
    df_train.to_csv("./exp_data/train.csv", index=False)
    df_val.to_csv("./exp_data/val.csv", index=False)
    df_test.to_csv("./exp_data/test.csv", index=False)




if __name__ == '__main__':

    # # main entry
    # parser = argparse.ArgumentParser()
    #
    # # general
    # parser.add_argument('--seed', type=int, default=0)
    #
    # # directory structure
    # parser.add_argument('--data_path', type=str, default='./exp_data/TOF1.csv')
    # parser.add_argument('--output_dir', type=str, default='results/segment/1')
    #
    # # mesh+prompt info
    # parser.add_argument('--training_part', type=float, default='0.7')
    # parser.add_argument('--object', nargs=1, default='cow')
    # parser.add_argument('--classes', nargs="+", default='sphere cube')


    # loading the data and save basic info
    data_path = './exp_data/TOF1.CSV'
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    info = {"exp": file_name}
    df = pd.read_csv(data_path, header=None)
    print("DF shape:", df.shape)

    #slice the data set into training ,validation, ignored part
    training_part = 0.7
    validing_part = 0.15
    slice_dataset(len(df), training_part, validing_part)
    info['slicing'] = f'{training_part}'

    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_ir('./exp_data', "./exp_data/train.csv", "./exp_data/val.csv", "./exp_data/test.csv")  # 应返回 numpy 或 tf.Tensor

    # create windows, the default is 20
    X_train, Y_train = create_dataset_window(X_train, Y_train)
    X_val, Y_val = create_dataset_window(X_val, Y_val)
    X_test, Y_test = create_dataset_window(X_test, Y_test)
    print(f"x_train shape: {X_train.shape},y_train shape: {Y_train.shape}")
    print(f"x_val shape: {X_val.shape},y_val shape: {Y_val.shape}")
    print(f"x_test shape: {X_test.shape},y_test: {Y_test.shape}")

    # create model
    model = tm.model_TCN_simple(hidden=3, num_filters=32, k_size=5, dense=32)
    info['model'] = 'simple_tcn'
    #model = tm.model_TCN_complete(hidden=3, num_filters=64, k_size=5, dense=64)

    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[rmse, 'mae'])
    info['loss'] = 'mse'
    history = model.fit(X_train, Y_train, epochs=2000, batch_size=15, validation_data=(X_val, Y_val))

    # test_scores = model.evaluate(X_test, Y_test, verbose=2)
    # print("Test loss:", test_scores[0])
    # print("Test mae:", test_scores[1])

    # call the model to predict
    output = model(X_test)
    # print(output[:3, :])
    # print("Output shape:", output.shape)

    # draw trajectory and loss of result and save
    model.summary()
    trajectory_curve(output, Y_test, info)
    loss_curve(history, info)