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

def trajectory_curve(labels, GT_value, args=None):

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

    # save results
    exp = os.path.splitext(os.path.basename(args.data_path))[0]
    save_path = os.path.join(args.output_dir, exp, "images")
    os.makedirs(save_path, exist_ok=True)
    filename = (f"{args.model}-"
                f"{args.folds}-"
                f"{args.training_part}-"
                f"{args.time_windows}-"
                f"{args.hidden}-"
                f"{args.num_filters}-"
                f"{args.kernel_size}-"
                f"{args.dense}-trajectory_comparison.png")
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=600)
    plt.close()


def loss_curve(history, args=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save results
    exp = os.path.splitext(os.path.basename(args.data_path))[0]
    save_path = os.path.join(args.output_dir, exp, "images")
    os.makedirs(save_path, exist_ok=True)
    filename = (f"{args.model}-"
                f"{args.folds}-"
                f"{args.training_part}-"
                f"{args.time_windows}-"
                f"{args.hidden}-"
                f"{args.num_filters}-"
                f"{args.kernel_size}-"
                f"{args.dense}-loss_comparison.png")
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=600)
    plt.close()

def metrics_curve(history, args=None):
    rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    plt.figure(figsize=(10, 6))

    # RMSE 曲线
    plt.plot(rmse, 'b-', label='Training RMSE')
    plt.plot(val_rmse, 'r-', label='Validation RMSE')

    # MAE 曲线
    plt.plot(mae, 'b--', label='Training MAE')
    plt.plot(val_mae, 'r--', label='Validation MAE')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    exp = os.path.splitext(os.path.basename(args.data_path))[0]
    save_path = os.path.join(args.output_dir, exp, "images")
    os.makedirs(save_path, exist_ok=True)
    filename = (f"{args.model}-"
                f"{args.folds}-"
                f"{args.training_part}-"
                f"{args.time_windows}-"
                f"{args.hidden}-"
                f"{args.num_filters}-"
                f"{args.kernel_size}-"
                f"{args.dense}-metrics.png")
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=600)
    plt.close()

def slice_dataset(df, training_part, validing_part, running_data_dir, args=None):
    num_data = len(df)
    folds =args.folds
    train_len = int(training_part * num_data)
    val_len = int(validing_part * num_data)
    ignore_len = num_data - train_len - val_len

    segments = {"train": train_len, "val": val_len, "ignore": ignore_len}
    orders = {
        1: ["train", "val", "ignore"],
        2: ["train", "ignore", "val"],
        3: ["val", "train", "ignore"],
        4: ["val", "ignore", "train"],
        5: ["ignore", "train", "val"],
        6: ["ignore", "val", "train"],
    }
    if folds not in orders:
        raise ValueError("folds must be in [1..6]")
    order = orders[folds]

    idx = 0
    slices = {}
    ranges = {}  # record (start, stop)
    for key in order:
        L = segments[key]
        s = slice(idx, idx + L)
        slices[key] = s
        ranges[key] = (s.start, s.stop)
        idx += L
    assert idx == num_data, "the sum of lengths of slices is not equal total length"

    df_train = df.iloc[slices["train"]]
    df_val = df.iloc[slices["val"]]
    df_ignore = df.iloc[slices["ignore"]]

    # print index info
    print(f"[fold={folds}] total={num_data} | train={len(df_train)} | val={len(df_val)} | ignore={len(df_ignore)}")
    for k in ("train", "val", "ignore"):
        s, e = ranges[k]
        print(f"  {k:<6} idx range: [{s}, {e}) len={e - s}")

    # ---- save data ----
    exp = os.path.splitext(os.path.basename(args.data_path))[0]
    save_path = os.path.join(running_data_dir, exp, str(folds))
    os.makedirs(save_path, exist_ok=True)
    df_train.to_csv(os.path.join(save_path, "train.csv"), index=False)
    df_val.to_csv(os.path.join(save_path, "val.csv"), index=False)
    df_ignore.to_csv(os.path.join(save_path, "test.csv"), index=False)

    # save path, slice, info etc
    return {
        "save_path": save_path,
        "slices": slices,         # {'train': slice(...), 'val': ..., 'ignore': ...}
        "ranges": ranges,         # {'train': (start, stop), ...}
        "lengths": segments,      # {'train': L, 'val': L, 'ignore': L}
    }


def slice_dataset2(df, training_part, validing_part, running_data_dir, args=None):
    ''' try to slice dataset into n folds '''
    # slide the data into three part
    num_data = len(df)
    folds = args.folds
    part = int(num_data/folds)


    train_len = int((folds-2)*part)
    val_len = part
    ignore_len = num_data - train_len - val_len

    if folds == 1:
        train_start = 0
        val_start = int(training_part * num_data)
        ignore_start = int((training_part + validing_part) * num_data)

    df_train = df.iloc[train_start:val_start]
    df_val = df.iloc[val_start:ignore_start]
    df_test = df.iloc[ignore_start:num_data]
    # 保存为新的 CSV 文件
    save_path = os.path.join(running_data_dir, str(folds))
    os.makedirs(save_path, exist_ok=True)
    df_train.to_csv(os.path.join(save_path, "train.csv"), index=False)
    df_val.to_csv(os.path.join(save_path, "val.csv"), index=False)
    df_test.to_csv(os.path.join(save_path, "test.csv"), index=False)
    return save_path

def optimazter(args):
    # create folders which used for results and temp files
    for path in [args.output_dir, args.running_data_dir]:
        os.makedirs(path, exist_ok=True)

    # loading the data and save basic info
    df = pd.read_csv(args.data_path, header=None)
    print("The whole dataset shape is:", df.shape)

    val_losses = []

    for fold in range(1, 7):
        print(f"\n===== Training Fold {fold}/6 =====")
        args.folds = fold

        # slice the data set into training(0.7) ,validation(0.15), ignored part(0.15)
        res = slice_dataset(df, args.training_part, args.validing_part, args.running_data_dir, args)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_ir(res["save_path"], feature=64)

        # create windows
        X_train, Y_train = create_dataset_window(X_train, Y_train, window_size=args.time_windows)
        X_val, Y_val = create_dataset_window(X_val, Y_val, window_size=args.time_windows)
        X_test, Y_test = create_dataset_window(X_test, Y_test, window_size=args.time_windows)

        # create model
        if args.model == "simple":
            model = tm.model_TCN_simple(hidden=args.hidden, num_filters=args.num_filters,
                                        k_size=args.kernel_size, dense=args.dense)
        elif args.model == "complete":
            model = tm.model_TCN_complete(hidden=args.hidden, num_filters=args.num_filters,
                                          k_size=args.kernel_size, dense=args.dense)

        # compile
        model.compile(optimizer=args.optimizer, loss=args.loss, metrics=args.metrics)

        # train
        history = model.fit(
            X_train, Y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, Y_val),
            verbose=1
        )

        # 保存该fold最小的val_loss
        min_val_loss = min(history.history['val_loss'])
        val_losses.append(min_val_loss)
        print(f"Fold {fold} best val_loss = {min_val_loss:.4f}")

        # ====== 计算平均和标准差 ======
    val_losses = np.array(val_losses)
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses, ddof=1)

    print("\n====== Cross Validation Results ======")
    print("Validation Losses (6 folds):", val_losses)
    print(f"Mean Validation Loss: {mean_loss:.4f}")
    print(f"STD Validation Loss: {std_loss:.4f}")

    exp = os.path.splitext(os.path.basename(args.data_path))[0]
    result_path = os.path.join(args.output_dir, exp, "cross_val_results.txt")
    with open(result_path, "w") as f:
        f.write("====== Cross Validation Results ======\n")
        f.write(f"Validation Losses (6 folds): {val_losses.tolist()}\n")
        f.write(f"Mean Validation Loss: {mean_loss:.4f}\n")
        f.write(f"STD Validation Loss: {std_loss:.4f}\n")

    print(f"\nResults saved to {result_path}")

    # # slice the data set into training(0.7) ,validation(0.15), ignored part(0.15)
    # res = slice_dataset(df, args.training_part, args.validing_part, args.running_data_dir, args)
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_ir(res["save_path"], feature=64)
    #
    # # create windows, the default is 20
    # X_train, Y_train = create_dataset_window(X_train, Y_train, window_size=args.time_windows)
    # X_val, Y_val = create_dataset_window(X_val, Y_val, window_size=args.time_windows)
    # X_test, Y_test = create_dataset_window(X_test, Y_test, window_size=args.time_windows)
    # print(f"The training part shape is: {X_train.shape}, its corresponding labels: {Y_train.shape}")
    # print(f"The validtion part shape is: {X_val.shape}, its corresponding labels: {Y_val.shape}")
    # print(f"The test part shape is : {X_test.shape}, its corresponding labels: {Y_test.shape}")
    #
    # # create model
    # if args.model == "simple":
    #     model = tm.model_TCN_simple(hidden=args.hidden, num_filters=args.num_filters, k_size=args.kernel_size, dense=args.dense)
    # elif args.model == "complete":
    #     model = tm.model_TCN_complete(hidden=args.hidden, num_filters=args.num_filters, k_size=args.kernel_size, dense=args.dense)
    #
    #
    # # compile the model
    # model.compile(optimizer=args.optimizer, loss=args.loss, metrics=args.metrics)
    #
    # # model trainning
    # history = model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, Y_val))
    #
    # # test_scores = model.evaluate(X_test, Y_test, verbose=2)
    # # print("Test loss:", test_scores[0])
    # # print("Test mae:", test_scores[1])
    #
    # # call the model to predict
    # output = model(X_test)
    # # print(output[:3, :])
    # # print("Output shape:", output.shape)
    #
    # # draw trajectory and loss of result and save
    # model.summary()
    # trajectory_curve(output, Y_test, args)
    # loss_curve(history, args)
    # metrics_curve(history, args)


def test(args):
    df = pd.read_csv(args.data_path, header=None)
    print("The whole dataset shape is:", df.shape)

    # slice the data set into training(0.7) ,validation(0.15), ignored part(0.15)
    res = slice_dataset(df, args.training_part, args.validing_part, args.running_data_dir, args)
    print(res["save_path"])




if __name__ == '__main__':

    # main entry
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=0)

    # directory structure
    parser.add_argument('--data_path', type=str, default='./exp_data/std_TOFEXP3.csv')
    parser.add_argument('--output_dir', type=str, default='results/segment/one')
    parser.add_argument('--running_data_dir', type=str, default='temp/')
    parser.add_argument('--folds', type=int, default='1')

    # basic info
    parser.add_argument('--training_part', type=float, default='0.7')
    parser.add_argument('--validing_part', type=float, default='0.15')
    parser.add_argument('--time_windows', type=int, default='20')

    #model info
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--hidden', type=int, default='3')
    parser.add_argument('--num_filters', type=int, default='32')
    parser.add_argument('--kernel_size', type=int, default='5')
    parser.add_argument('--dense', type=int, default='32')
    parser.add_argument('--epochs', type=int, default='1')
    parser.add_argument('--batch_size', type=int, default='64')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--metrics', nargs='+', type=str, default=[rmse, 'mae'])
    parser.add_argument('--dropout_rate', type=float, default='0.005')

    args = parser.parse_args()
    optimazter(args)

    # test(args)



