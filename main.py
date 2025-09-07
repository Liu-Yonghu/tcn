import os
import json
import utils
import math
from tcn_old import TCN
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import matplotlib
import keras_tuner
from keras import layers, models
import tensorflow as tf
from keras.callbacks import TensorBoard,Callback
import datetime

matplotlib.use("TkAgg")
def create_dataset_window(X_all, Y_all, window_size=20):
    X_list = []
    Y_list = []

    for i in range(len(X_all) - window_size + 1):
        X_seq = X_all[i: i + window_size,:]        # shape (20, 64)
        #Y_target = Y_all[i + window_size-1]
        Y_target = Y_all[i + np.floor(window_size / 2).astype(int) + 1,:]     # from  middle smaples

        X_list.append(X_seq)
        Y_list.append(Y_target)

    X = np.stack(X_list)  # (N, 15, 64)
    Y = np.stack(Y_list)  # (N, 2)
    return X, Y

def create_tf_dataset(data_array, output_array, input_sequence_length=20, output_sequence_length=1, batch_size=1, shuffle=False, multi_horizon=False,):
    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-2, :], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = np.floor(input_sequence_length / 2).astype(int) + 1
    target_offset2 = math.floor(input_sequence_length / 2) + 1
    print(f"target_offset:{target_offset}")
    print(f"target_offset2:{target_offset2}")
    target_seq_length = output_sequence_length
    targets = timeseries_dataset_from_array(
        output_array[target_offset:-target_offset, :],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset

def trajectory_curve(labels, GT_value, args=None):
    X_T = GT_value[:, :1]
    Y_T = GT_value[:, 1:]
    X_P = labels[:, :1]
    Y_P= labels[:, 1:]
    # create figure
    plt.figure(figsize=(6, 6))
    plt.plot(X_P, Y_P, marker='x', linestyle='-', color='blue', label='Prediction')
    plt.plot(X_T, Y_T, marker='o', linestyle='--', color='red', label='Ground True')
    plt.xlim(0, 3)
    plt.ylim(0, 3)

    # set title and label
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

    # RMSE curve
    plt.plot(rmse, 'b-', label='Training RMSE')
    plt.plot(val_rmse, 'r-', label='Validation RMSE')

    # MAE curve
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

def optimizer(args):
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

        # test or ignore
        X_test, Y_test = create_dataset_window(X_test, Y_test, window_size=args.time_windows)

        # create model
        model = 0
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
        # draw trajectory and loss of result and save
        model.summary()
        # output = model(X_test)
        # trajectory_curve(output, Y_test, args)
        loss_curve(history, args)
        metrics_curve(history, args)

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


#==========================================simple keras===============================================
# simple keras automate search, the more complex contrl of autokeras, reference NAS_2.py or NAS_3.py
def build_model(hp, args):
    if args.model == "simple":
        model = tm.model_TCN_simple(
            hidden=hp.Choice("hidden", [2, 3, 4]),
            num_filters=hp.Choice("nb_filters", [8, 16, 32]),
            k_size=hp.Choice("k_size", [2, 3, 4, 5]),
            dense=hp.Choice("dense", [8, 16, 32])
        )
    elif args.model == "complete":
        model = tm.model_TCN_complete(
            hidden=hp.Choice("hidden", [2, 3, 4]),
            num_filters=hp.Choice("nb_filters", [8, 16, 32]),
            k_size=hp.Choice("k_size", [2, 3, 4, 5]),
            dense=hp.Choice("dense", [8, 16, 32])
        )

    model.compile(
        optimizer=args.optimizer,
        loss=args.loss,
        metrics=args.metrics
    )
    return model
def prepare_data(args):
    df = pd.read_csv(args.data_path, header=None)
    print("The whole dataset shape is:", df.shape)

    res = slice_dataset(df, args.training_part, args.validing_part, args.running_data_dir, args)
    print("data partition saved in:", res["save_path"])

    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_ir(res["save_path"], feature=64)

    X_train, Y_train = create_dataset_window(X_train, Y_train, window_size=args.time_windows)
    X_val, Y_val = create_dataset_window(X_val, Y_val, window_size=args.time_windows)
    X_test, Y_test = create_dataset_window(X_test, Y_test, window_size=args.time_windows)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
class LossPlotter(Callback):
    def __init__(self, save_dir, trial_id, execution_id):
        super().__init__()
        self.save_dir = save_dir
        self.trial_id = trial_id
        self.execution_id = execution_id
        self.history = {"loss": [], "val_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))

        plt.figure()
        plt.plot(self.history["loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Trial {self.trial_id} - Exec {self.execution_id} Loss Curve")
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(f"{self.save_dir}/trial_{self.trial_id}/exec_{self.execution_id}_loss.png")
        plt.close()
class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        original_callbacks = kwargs.pop("callbacks", [])
        histories = []
        for execution in range(self.executions_per_trial):
            callbacks = original_callbacks[:]
            # every execution has one LossPlotter
            loss_plotter = LossPlotter(
                save_dir=self.project_dir,
                trial_id=trial.trial_id,
                execution_id=execution
            )
            callbacks.append(loss_plotter)
            kwargs["callbacks"] = callbacks
            history = super().run_trial(trial, *args, **kwargs)
            histories.append(history)
        return histories
def run_search(args, max_trials=10, executions_per_trial=10, epochs=100, batch_size=32):

    (X_train, Y_train), (X_val, Y_val), _ = prepare_data(args)

    exp = os.path.splitext(os.path.basename(args.data_path))[0]
    save_path = f"results/tcn_search/{str(exp)}"

    tuner = MyTuner(
        hypermodel=lambda hp: build_model(hp, args),
        objective="val_loss",
        max_trials=max_trials,
        seed=args.seed,
        executions_per_trial=executions_per_trial,
        directory=".",
        project_name=save_path
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(filepath=f"{save_path}/best_model.h5", monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    #reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

    # log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    tuner.search(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    best_val_loss = best_trial.score
    best_trial_id = best_trial.trial_id

    print("Best hyperparameters found:", best_hps.values)
    print(f"Best trial id: {best_trial_id}, best val_loss: {best_val_loss:.4f}")

    results_file = os.path.join(save_path, "best_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "best_hyperparameters": best_hps.values,
            "best_val_loss": best_val_loss,
            "best_trial_id": best_trial_id,
            "project": save_path
        }, f, indent=4)

    print(f"Best hyperparameters saved to {results_file}")
    return best_model, best_hps, tuner, save_path
def collect_tuner_results(tuner_dir):
    results = []
    # read oracle.json，the score and index of trial
    oracle_path = os.path.join(tuner_dir, "oracle.json")
    if os.path.exists(oracle_path):
        with open(oracle_path, "r") as f:
            oracle = json.load(f)
        trials = oracle.get("display").get("trial_number", {"None"})
        print(trials)
    else:
        print("oracle.json not found, try to read in trial_x folder")
        trials = {}

    # search all trial_x folders
    for trial_id in os.listdir(tuner_dir):
        trial_dir = os.path.join(tuner_dir, trial_id)
        trial_file = os.path.join(trial_dir, "trial.json")

        if os.path.isdir(trial_dir) and os.path.exists(trial_file):
            with open(trial_file, "r") as f:
                trial_data = json.load(f)

            trial_info = {
                "trial_id": trial_id,
                "score": trial_data.get("score", None),  # val_loss
            }
            # find hyperparmeter
            hp = trial_data.get("hyperparameters", {}).get("values", {})
            trial_info.update(hp)
            results.append(trial_info)

    # turn into DataFrame
    df = pd.DataFrame(results)

    if not df.empty:
        # find the optimal trial（val_loss min）
        best_idx = df["score"].astype(float).idxmin()
        df["is_best"] = False
        df.loc[best_idx, "is_best"] = True

        best_trial = df.loc[best_idx].to_dict()
        print("Optimal Trial:")
        print(f"Trial ID: {best_trial['trial_id']}")
        print(f"Score (val_loss): {best_trial['score']}")
        print("The best parameter combination:")
        for k, v in best_trial.items():
            if k not in ["trial_id", "score", "is_best"]:
                print(f"  {k}: {v}")
    else:
        df["is_best"] = []

    output_path = os.path.join(tuner_dir, "tuner_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"The results saved on {output_path}")
    return df

if __name__ == '__main__':

    # main entry
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=777)

    # directory structure
    parser.add_argument('--data_path', type=str, default='./exp_data/std_TOFEXP1.csv')
    parser.add_argument('--output_dir', type=str, default='results/normal_train/one')
    parser.add_argument('--running_data_dir', type=str, default='temp/')
    parser.add_argument('--folds', type=int, default='1')

    # basic info
    parser.add_argument('--training_part', type=float, default='0.7')
    parser.add_argument('--validing_part', type=float, default='0.15')
    parser.add_argument('--time_windows', type=int, default='20')

    #model info
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--hidden', type=int, default='4')
    parser.add_argument('--num_filters', type=int, default='16')
    parser.add_argument('--kernel_size', type=int, default='5')
    parser.add_argument('--dense', type=int, default='8')
    parser.add_argument('--epochs', type=int, default='20')
    parser.add_argument('--batch_size', type=int, default='32')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--metrics', nargs='+', type=str, default=[rmse, 'mae'])
    parser.add_argument('--dropout_rate', type=float, default='0.005')

    args = parser.parse_args()

    ## optimizer is the normal training function, which has fixed parameter
    # optimizer(args)

    best_model, best_hps, tuner, results_path = run_search(args, max_trials=20, executions_per_trial=3, epochs=500)
    collect_tuner_results(results_path)







