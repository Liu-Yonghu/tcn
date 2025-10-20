import os
import json
import utils
import pandas as pd
from math import inf
import numpy as np
import tcn_model as tm
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import argparse
from utils import rmse, create_tf_dataset, create_dataset_window, compute_sparc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras_tuner
from keras.callbacks import TensorBoard, Callback



# matplotlib.use("Agg")
def trajectory_curve(labels, GT_value, args=None):
    X_T = GT_value[:, :1]
    Y_T = GT_value[:, 1:]
    X_P = labels[:, :1]
    Y_P = labels[:, 1:]
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
    folds = args.folds
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
    df_train.to_csv(os.path.join(save_path, "train.csv"), header=False, index=False)
    df_val.to_csv(os.path.join(save_path, "val.csv"), header=False, index=False)
    df_ignore.to_csv(os.path.join(save_path, "test.csv"), header=False, index=False)

    # save path, slice, info etc
    return {
        "save_path": save_path,
        "slices": slices,  # {'train': slice(...), 'val': ..., 'ignore': ...}
        "ranges": ranges,  # {'train': (start, stop), ...}
        "lengths": segments,  # {'train': L, 'val': L, 'ignore': L}
    }
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

        if args.data_source == "ir":
            X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_ir(res["save_path"], feature=64)
        elif args.data_source == "mmwave":
            X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_mmwave(res["save_path"], feature=3)
        else:
            raise ValueError(f"Unsupported data source: {args.data_source}")

        # create windows
        X_train, Y_train = create_dataset_window(X_train, Y_train, window_size=args.time_windows)
        X_val, Y_val = create_dataset_window(X_val, Y_val, window_size=args.time_windows)

        # test or ignore
        X_test, Y_test = create_dataset_window(X_test, Y_test, window_size=args.time_windows)

        # (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = prepare_data(args)

        print("Train X range:", Y_train[0].min(), Y_train[0].max())
        print("Train X range:", Y_train[1].min(), Y_train[1].max())
        print("Val X range:", Y_val[0].min(), Y_val[0].max())
        print("Val Y range:", Y_val[1].min(), Y_val[1].max())

        # create model
        model = 0
        if args.model == "simple":
            model = tm.model_TCN_simple(hidden=args.hidden, num_filters=args.num_filters,
                                        k_size=args.kernel_size, dense=args.dense, source=args.data_source)
        elif args.model == "complete":
            model = tm.model_TCN_complete(hidden=args.hidden, num_filters=args.num_filters,
                                          k_size=args.kernel_size, dense=args.dense, source=args.data_source)

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

        # 保存该fold mean 的val_loss
        mean_val_loss = np.mean(history.history['val_loss'])
        val_losses.append(mean_val_loss)
        mean_test_loss = np.mean(model.evaluate(X_test, Y_test, verbose=0))
        print(f"Fold {fold} test_loss = {mean_test_loss:.4f}")
        print(f"Fold {fold} mean val_loss = {mean_val_loss:.4f}")

    # ====== calculate mean and standard deviation of 6 folds ======
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


# ==========================================simple keras===============================================
# simple keras automate search, the more complex contrl of autokeras, reference NAS_2.py or NAS_3.py
def build_model(hp, args):
    model = None
    if args.model == "simple":
        model = tm.model_TCN_simple(
            hidden=hp.Choice("hidden", [2, 3, 4]),
            num_filters=hp.Choice("nb_filters", [8, 16, 32]),
            k_size=hp.Choice("k_size", [2, 3, 4, 5]),
            dense=hp.Choice("dense", [8, 16, 32]),
            source=args.data_source
        )
    elif args.model == "complete":
        model = tm.model_TCN_complete(
            hidden=hp.Choice("hidden", [2, 3, 4]),
            num_filters=hp.Choice("nb_filters", [8, 16, 32]),
            k_size=hp.Choice("k_size", [2, 3, 4, 5]),
            dense=hp.Choice("dense", [8, 16, 32]),
            source=args.data_source
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

    if args.data_source == "ir":
        X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_ir(res["save_path"], feature=64)
    elif args.data_source == "mmwave":
        X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_dataset_mmwave(res["save_path"], feature=3)
    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")

    input_sequence_length, output_sequence_length, batch_size = args.time_windows, 1, args.batch_size
    train_ds = create_tf_dataset(X_train, Y_train, input_sequence_length, output_sequence_length, batch_size)
    val_ds = create_tf_dataset(X_val, Y_val, input_sequence_length, output_sequence_length, batch_size)
    test_ds = create_tf_dataset(X_test, Y_test, input_sequence_length, output_sequence_length, batch_size)
    # for plot evaluate figure
    all_train_x = []
    all_train_y = []

    all_val_x = []
    all_val_y = []

    all_test_x = []
    all_test_y = []

    for batch_x, batch_y in train_ds:
        all_train_x.append(batch_x.numpy())  # Convert to numpy array
        all_train_y.append(batch_y.numpy())  # Convert to numpy array

    train_x = np.concatenate(all_train_x, axis=0)
    train_y = np.concatenate(all_train_y, axis=0)


    for batch_x, batch_y in val_ds:
        all_val_x.append(batch_x.numpy())  # Convert to numpy array
        all_val_y.append(batch_y.numpy())  # Convert to numpy array

    val_x = np.concatenate(all_val_x, axis=0)
    val_y = np.concatenate(all_val_y, axis=0)


    for batch_x, batch_y in test_ds:
        all_test_x.append(batch_x.numpy())  # Convert to numpy array
        all_test_y.append(batch_y.numpy())  # Convert to numpy array

    test_x = np.concatenate(all_test_x, axis=0)
    test_y = np.concatenate(all_test_y, axis=0)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


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


class TrialExecutionLogger(Callback):
    def __init__(self, trial_id, execution_id, max_epochs):
        super().__init__()
        self.trial_id = trial_id
        self.execution_id = execution_id
        self.max_epochs = max_epochs

    def on_train_begin(self, logs=None):
        print(f"\n[Trial {self.trial_id} | Exec {self.execution_id}] "
              f"Start training (max_epochs={self.max_epochs})")

    def on_epoch_end(self, epoch, logs=None):
        print(f"[Trial {self.trial_id} | Exec {self.execution_id}] "
              f"Epoch {epoch + 1}/{self.max_epochs}, "
              f"loss={logs.get('loss'):.4f}, val_loss={logs.get('val_loss'):.4f}")

    def on_train_end(self, logs=None):
        print(f"[Trial {self.trial_id} | Exec {self.execution_id}] Training finished\n")

class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        original_callbacks = kwargs.pop("callbacks", [])
        histories = []
        for execution in range(self.executions_per_trial):
            print(f"\n=== Starting Trial {trial.trial_id}, Execution {execution} ===")

            callbacks = original_callbacks[:]

            loss_plotter = LossPlotter(
                save_dir=self.project_dir,
                trial_id=trial.trial_id,
                execution_id=execution
            )
            callbacks.append(loss_plotter)

            callbacks.append(TrialExecutionLogger(
                trial_id=trial.trial_id,
                execution_id=execution,
                max_epochs=kwargs.get("epochs", 0)
            ))

            kwargs["callbacks"] = callbacks
            history = super().run_trial(trial, *args, **kwargs)
            histories.append(history)

            print(f"=== Finished Trial {trial.trial_id}, Execution {execution} ===\n")

        return histories


def run_search(args, max_trials=10, executions_per_trial=10, epochs=100, batch_size=32):
    (X_train, Y_train), (X_val, Y_val), _ = prepare_data(args)

    exp = os.path.splitext(os.path.basename(args.data_path))[0]
    save_path = os.path.join("results", "tcn_search", exp)
    os.makedirs(save_path, exist_ok=True)

    tuner = MyTuner(
        hypermodel=lambda hp: build_model(hp, args),
        objective=keras_tuner.Objective("val_loss", "min"),
        max_trials=max_trials,
        seed=args.seed,
        executions_per_trial=executions_per_trial,
        directory="results/tcn_search",
        project_name=exp
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path, "best_model.keras"),
        monitor="val_loss", save_weights_only=True,
        save_best_only=True, verbose=1
    )

    print(f"Starting hyperparameter search for experiment: {exp}")

    tuner.search(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=0  # 日志由 TrialExecutionLogger 打印
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
    parser.add_argument('--seed', type=int, default=42)

    # directory structure
    parser.add_argument('--data_source', type=str, choices=['ir', 'mmwave'], default='mmwave')
    parser.add_argument('--data_path', type=str, default='./exp_data/norm_mmWaveEXP3.csv') #datasets/preprocessed-RadarEXP1.csv_
    parser.add_argument('--output_dir', type=str, default='results/normal_train/one')
    parser.add_argument('--running_data_dir', type=str, default='temp/')
    parser.add_argument('--folds', type=int, default='1')

    # basic info
    parser.add_argument('--training_part', type=float, default='0.7')
    parser.add_argument('--validing_part', type=float, default='0.15')
    parser.add_argument('--time_windows', type=int, default='12')

    # model info
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--hidden', type=int, default='2')
    parser.add_argument('--num_filters', type=int, default='8')
    parser.add_argument('--kernel_size', type=int, default='2')
    parser.add_argument('--dense', type=int, default='32')
    parser.add_argument('--epochs', type=int, default='50')
    parser.add_argument('--batch_size', type=int, default='32')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--metrics', nargs='+', type=str, default=[rmse, 'mae'])
    parser.add_argument('--dropout_rate', type=float, default='0.01')

    args = parser.parse_args()

    # optimizer is the normal training function, which has fixed parameter
    #optimizer(args)
    #

    best_model, best_hps, tuner, results_path = run_search(args, max_trials=10, executions_per_trial=10, epochs=200)
    collect_tuner_results(results_path)


    ## pre-prepare dataset for remote server training
    # df = pd.read_csv(args.data_path, header=None)
    # print("The whole dataset shape is:", df.shape)
    #
    # for fold in range(1, 7):
    #     print(f"\n===== Training Fold {fold}/6 =====")
    #     args.folds = fold
    #
    #     # slice the data set into training(0.7) ,validation(0.15), ignored part(0.15)
    #     (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = prepare_data(args)
    #
    #     print("Train X range:", Y_train[0].min(), Y_train[0].max())
    #     print("Train Y range:", Y_train[1].min(), Y_train[1].max())
    #     print("Val X range:", Y_val[0].min(), Y_val[0].max())
    #     print("Val Y range:", Y_val[1].min(), Y_val[1].max())
