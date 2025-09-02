import os
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
import json
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os
import pathlib
import pandas as pd
import sys
import argparse
import glob
from tensorflow import keras
import keras_tuner
from tensorboard.plugins.hparams import api as hparams_api
import copy
from tensorflow.keras import regularizers, optimizers, losses, initializers, metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from tensorflow.keras.layers import Input, TimeDistributed, Concatenate, Lambda
from tensorflow.keras import backend as K
#from tcn import TCN_model
from tcn_old import TCN
from keras_tuner import backend
from keras_tuner import errors
from keras_tuner import utils
from keras_tuner.engine import base_tuner
from keras_tuner.engine import tuner_utils
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from scipy import stats

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("NO gpu available")

# param settings
batch_size = 32
epochs = 3000
directory = '.'
input_sequence_length = 20
output_sequence_length = 1

# Parent Directory path
parent_dir = "temp/TOF1/1"

# Path
path = directory

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory '%s' created" % directory)


# loading Dataset#
def load_dataset(path, file_training, file_valid, file_testing):
    training_data = file_training
    validation_data = file_valid
    testing_data = file_testing
    trainList = list()
    valList = list()
    testList = list()

    with open(training_data, 'r') as train_inp_csv:
        train_inp_csv_reader = csv.reader(train_inp_csv)
        for rowTr in train_inp_csv_reader:
            trainList.append(rowTr)

    trainArray = np.asarray(trainList, dtype=np.float32)
    # print(trainArray.shape)
    train_X = trainArray[:, 0:64]
    train_Y = trainArray[:, 64:]
    #####################################

    with open(validation_data, 'r') as val_inp_csv:
        val_inp_csv_reader = csv.reader(val_inp_csv)
        for rowVl in val_inp_csv_reader:
            valList.append(rowVl)

    valArray = np.asarray(valList, dtype=np.float32)
    # print(valArray.shape)

    val_X = valArray[:, 0:64]
    val_Y = valArray[:, 64:]
    #####################################

    with open(testing_data, 'r') as test_inp_csv:
        test_inp_csv_reader = csv.reader(test_inp_csv)
        for rowTe in test_inp_csv_reader:
            testList.append(rowTe)

    testArray = np.asarray(testList, dtype=np.float32)
    # print(valArray.shape)

    test_X = testArray[:, 0:64]
    test_Y = testArray[:, 64:]

    X_train = np.array(train_X)  # np.transpose(train_X)
    Y_train = np.array(train_Y)  # np.transpose(train_Y)

    X_val = np.array(val_X)  # np.transpose(test1_X)
    Y_val = np.array(val_Y)  # np.transpose(test1_Y)

    X_test = np.array(test_X)  # np.transpose(test2_X)
    Y_test = np.array(test_Y)  # np.transpose(test2_Y)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def create_tf_dataset(data_array, output_array, input_sequence_length, output_sequence_length, batch_size=1, shuffle=False):
    inputs = timeseries_dataset_from_array(
        data_array[:-2, :],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = math.floor(input_sequence_length / 2) + 1
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


X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(parent_dir, "temp/TOF1/1/train.csv",
                                                              "temp/TOF1/1/val.csv",
                                                              "temp/TOF1/1/test.csv")
train_ds = create_tf_dataset(X_train, Y_train, input_sequence_length, output_sequence_length, batch_size)
val_ds = create_tf_dataset(X_val, Y_val, input_sequence_length, output_sequence_length, batch_size)
test_ds = create_tf_dataset(X_test, Y_test, input_sequence_length, output_sequence_length, batch_size)


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


class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        # Supponiamo che mlp_model sia la tua MLP già allenata e tcn_model sia la TCN
        x = Input(shape=(20, 64))
        tcn_model = model_TCN(hp.Choice("hidden", [2, 3, 4]), hp.Choice("nb_filters", [8, 16, 32]),
                              hp.Choice("k_size", [2, 3, 4, 5]), hp.Choice("dense", [8, 16, 32]))
        # Inserisci la sequenza concatenata nella TCN
        tcn_output = tcn_model(x)
        print("tcn_out:", tcn_output.shape)
        full_model = models.Model(inputs=x, outputs=tcn_output)

        return full_model

    def fit(self, hp, model, trial, execution, train_ds, val_ds, test_ds, path_save, callbacks=None, **kwargs):

        loss_values = np.zeros(epochs)
        val_loss_values = np.zeros(epochs)
        train_acc_values = np.zeros(epochs)
        val_acc_values = np.zeros(epochs)
        best_epoch = 0
        best_val_loss = 100
        best_test_loss = 100
        s_loss = 0
        v_loss = 0

        # Average the loss across the batch size within an epoch
        student_loss_fn = losses.mean_squared_error
        # valid_loss = losses.mean_sqaured_error(name="valid_loss")

        # Specify the performance metric
        optimizer = tf.keras.optimizers.Adamax(learning_rate=0.0001)
        train_acc = tf.keras.metrics.MeanSquaredError()
        valid_acc = tf.keras.metrics.MeanSquaredError()
        test_acc = tf.keras.metrics.MeanSquaredError()

        @tf.function(jit_compile=True)
        def run_train_step(x, y, student_model):
            # Unpack data
            with tf.GradientTape() as tape:
                # Forward pass of student
                student_predictions = student_model(x, training=True)
                student_predictions = tf.squeeze(student_predictions)
                # print(y.shape)
                # print(student_predictions.shape)
                # Compute losses
                student_loss = tf.reduce_mean(student_loss_fn(y, student_predictions))

            # Compute gradients
            trainable_vars = student_model.trainable_weights
            gradients = tape.gradient(student_loss, trainable_vars)

            # Update weights
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update the metrics configured in `compile()`.
            train_acc.update_state(y, student_predictions)
            return (student_loss)

        @tf.function(jit_compile=True)
        def run_val_step(x, y, student_model):
            # Compute predictions
            y_prediction_val = student_model(x, training=False)
            y_prediction_val = tf.squeeze(y_prediction_val)
            val_loss = tf.reduce_mean(student_loss_fn(y, y_prediction_val))
            # Update the metrics.
            valid_acc.update_state(y, y_prediction_val)
            return val_loss

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model
            callback.on_train_begin(self)
        print("trial num:", trial.trial_id)
        print("exec num:", execution)

        for epoch in range(epochs):
            tot_s_loss = 0
            print("\nStart of epoch %d" % (epoch,))
            for step, (X_train, Y_train) in enumerate(train_ds):
                s_loss = run_train_step(X_train, Y_train, model)
                tot_s_loss = tot_s_loss + s_loss.numpy()
                # print(s_loss)

            tot_s_loss = tot_s_loss / (step + 1)
            # print("step:",(len(train_ds)))
            # print("student_loss_tot:",s_loss.numpy())
            loss_values[epoch] = tot_s_loss
            train_acc_values[epoch] = train_acc.result()
            # Reset training metrics at the end of each epoch
            train_acc.reset_states()

            tot_v_loss = 0
            for step_val, (X_val, Y_val) in enumerate(val_ds):
                v_loss = run_val_step(X_val, Y_val, model)
                tot_v_loss = tot_v_loss + v_loss.numpy()
                # print(step_val)

            tot_v_loss = tot_v_loss / (step_val + 1)

            # print("student_lossval_tot:",v_loss.numpy())
            val_loss_values[epoch] = tot_v_loss
            val_acc_values[epoch] = valid_acc.result()
            valid_acc.reset_states()

            # print("Validation loss epoch %d: %.4f" %(epoch, float(val_loss_values[epoch])))

            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"val_loss": val_loss_values[epoch], "loss": loss_values[epoch]})

            template = "Epoch {}, train_loss: {:.5f}, val_loss: {:.5f}"
            print(template.format(epoch,
                                  loss_values[epoch],
                                  val_loss_values[epoch]
                                  ))
            if epoch < 5:
                wait = 0
                best_epoch = epoch
                best_val_loss = val_loss_values[epoch]
            if val_loss_values[epoch] < best_val_loss and epoch >= 5:
                best_epoch = epoch
                wait = 0
                best_val_loss = val_loss_values[epoch]
            if val_loss_values[epoch] > best_val_loss and epoch > 5:
                wait += 1
                print("Epoch %03d val_loss did not improve" % (epoch))
            if wait >= 50:
                print("Early stopping")
                break

        epochs_num = range(1, len(loss_values) + 1)
        plt.figure()
        #
        # Plot the model accuracy vs Epochs
        #
        plt.plot(epochs_num[5:epoch], loss_values[5:epoch], 'r', label='Training loss')
        plt.plot(epochs_num[5:epoch], val_loss_values[5:epoch], 'b', label='Validation loss')
        plt.title('Training & Validation Loss', fontsize=16)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
        plt.savefig('./autokeras_mlp4tcn_res_exp2_all/trial_' + str(trial.trial_id) + '/learning_curve_mlptcn' + str(
            execution) + '.pdf')
        plt.close()

        test_x, test_y = zip(*test_ds)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        all_train_x = []
        all_train_y = []

        all_val_x = []
        all_val_y = []

        for batch_x, batch_y in train_ds:
            all_train_x.append(batch_x.numpy())  # Converti in numpy array
            all_train_y.append(batch_y.numpy())  # Converti in numpy array

        # Concatenare i batch per ottenere un array completo
        train_x = np.concatenate(all_train_x, axis=0)
        train_y = np.concatenate(all_train_y, axis=0)

        # val_x, val_y = zip(*val_ds)
        # val_x = np.array(val_x)
        # val_y = np.array(val_y)

        for batch_x, batch_y in val_ds:
            all_val_x.append(batch_x.numpy())  # Converti in numpy array
            all_val_y.append(batch_y.numpy())  # Converti in numpy array

        # Concatenare i batch per ottenere un array completo
        val_x = np.concatenate(all_val_x, axis=0)
        val_y = np.concatenate(all_val_y, axis=0)

        try:
            model.load_weights("./autokeras_mlp4tcn_res_exp2_all/trial_" + str(trial.trial_id) + "/ckpt_exec" + str(
                int(execution)) + ".h5")
            model.compile(optimizer='adam', loss='mse')
            # Valutare il modello
            best_test_loss = model.evaluate(test_x, test_y)
            print(f"Mean Squared Error on test data: {best_test_loss}")

            # Usare il modello per fare predizioni
            predictions = model.predict(test_x)

            # Visualizzare gli input, le distanze reali e le predizioni
            plt.figure(figsize=(12, 6))

            # Plot delle letture del sensore (input) vs distanza reale
            plt.scatter(test_y[:, 0], test_y[:, 1], color='blue', label='Distanze reali', alpha=0.5)
            plt.scatter(predictions[:, 0], predictions[:, 1], color='red', label='Distanze predette', alpha=0.5)
            plt.xlabel('Distanza y')
            plt.ylabel('Distanza x')
            plt.xlim(0, 3)
            plt.ylim(0, 3)
            plt.title('Posizione vera vs posizione predetta ')
            plt.grid()
            plt.legend()

            plt.tight_layout()
            plt.savefig('./autokeras_mlp4tcn_res_exp2_all/trial_' + str(trial.trial_id) + '/test_results_' + str(
                execution) + '.pdf')
            plt.close()
            # #################################
            # Usare il modello per fare predizioni train
            predictions_train = model.predict(train_x)

            # Visualizzare gli input, le distanze reali e le predizioni
            plt.figure(figsize=(12, 6))

            # Plot delle letture del sensore (input) vs distanza reale
            plt.scatter(train_y[:, 0], train_y[:, 1], color='blue', label='Distanze Reali', alpha=0.5)
            plt.scatter(predictions_train[:, 0], predictions_train[:, 1], color='red', label='Distanze predette',
                        alpha=0.5)
            plt.xlabel('Distanza y')
            plt.ylabel('Distanza x')
            plt.xlim(0, 3)
            plt.ylim(0, 3)
            plt.title('Posizione vera vs posizione predetta ')
            plt.grid()
            plt.legend()

            plt.tight_layout()
            plt.savefig('./autokeras_mlp4tcn_res_exp2_all/trial_' + str(trial.trial_id) + '/train_results_' + str(
                execution) + '.pdf')
            plt.close()

            #################################
            # Usare il modello per fare predizioni val
            predictions_val = model.predict(val_x)

            # Visualizzare gli input, le distanze reali e le predizioni
            plt.figure(figsize=(12, 6))

            # Plot delle letture del sensore (input) vs distanza reale
            plt.scatter(val_y[:, 0], val_y[:, 1], color='blue', label='Distanze Reali', alpha=0.5)
            plt.scatter(predictions_val[:, 0], predictions_val[:, 1], color='red', label='Distanze predette', alpha=0.5)
            plt.xlabel('Distanza y')
            plt.ylabel('Distanza x')
            plt.xlim(0, 3)
            plt.ylim(0, 3)
            plt.title('Posizione vera vs posizione predetta ')
            plt.grid()
            plt.legend()

            plt.tight_layout()
            plt.savefig('./autokeras_mlp4tcn_res_exp2_all/trial_' + str(trial.trial_id) + '/val_results_' + str(
                execution) + '.pdf')
            plt.close()


        except Exception as e:
            print("Errore durante la valutazione o la creazione dei plot:", e)

        # print("sonoqui3")

        for callback in callbacks:
            # The "my_metric" is the objective passed to the tuner.
            callback.on_train_end(
                logs={"best_val_loss": best_val_loss, "best_test_loss": best_test_loss, "best_epoch": best_epoch})

        return best_test_loss


# class  BayesianOptimization(keras_tuner.BayesianOptimization):
class GridSearchTuner(keras_tuner.GridSearch):
    def __init__(self, hypermodel, **kwargs):
        super().__init__(hypermodel, **kwargs)

    def _build_and_fit_model(self, trial, execution, *args, **kwargs):
        """For AutoKeras to override.

        DO NOT REMOVE this function. AutoKeras overrides the function to tune
        tf.data preprocessing pipelines, preprocess the dataset to obtain
        the input shape before building the model, adapt preprocessing layers,
        and tune other fit_args and fit_kwargs.

        Args:
            trial: A `Trial` instance that contains the information needed to
                run this trial. `Hyperparameters` can be accessed via
                `trial.hyperparameters`.
            *args: Positional arguments passed by `search`.
            **kwargs: Keyword arguments passed by `search`.

        Returns:
            The fit history.
        """
        hp = trial.hyperparameters
        model = self._try_build(hp)
        results = self.hypermodel.fit(hp, model, trial, execution, *args, **kwargs)

        # Save the build config for model loading later.
        if backend.config.multi_backend():
            utils.save_json(
                self._get_build_config_fname(trial.trial_id),
                model.get_build_config(),
            )

        tuner_utils.validate_trial_results(
            results, self.oracle.objective, "HyperModel.fit()"
        )
        return results

    def run_trial(self, trial, *args, **kwargs):
        original_callbacks = kwargs.pop("callbacks", [])

        for callback in original_callbacks:
            print(callback.__class__.__name__)

            # Run the training process multiple times.
        histories = []
        for execution in range(self.executions_per_trial):
            K.clear_session()
            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            checkpoint_file_name = self._get_checkpoint_fname(trial.trial_id, execution)
            print("checkpoint_file_path:", checkpoint_file_name)
            callback_check = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_file_name,
                monitor='val_loss',
                verbose=1,
                mode='min',
                save_best_only=True,
                save_weights_only=True)
            callbacks.append(callback_check)
            copied_kwargs["callbacks"] = callbacks
            obj_value = self._build_and_fit_model(trial, execution, *args, **copied_kwargs)
            histories.append(obj_value)
        return histories

    def on_batch_begin(self, trial, model, batch, logs):
        pass

    def on_batch_end(self, trial, model, batch, logs=None):
        pass

    def on_epoch_begin(self, trial, model, epoch, logs=None):
        pass

    def on_epoch_end(self, trial, model, epoch, logs=None):
        pass

    def get_best_models(self, num_models=1):
        return super().get_best_models(num_models)

    def _deepcopy_callbacks(self, callbacks):
        try:
            callbacks = copy.deepcopy(callbacks)
        except:
            raise keras_tuner.errors.FatalValueError(
                "All callbacks used during a search "
                "should be deep-copyable (since they are "
                "reused across trials). "
                "It is not possible to do `copy.deepcopy(%s)`" % (callbacks,)
            )
        return callbacks

    def _configure_tensorboard_dir(self, callbacks, trial, execution):
        for callback in callbacks:
            if callback.__class__.__name__ == "TensorBoard":
                # Patch TensorBoard log_dir and add HParams KerasCallback
                logdir = self._get_tensorboard_dir(
                    callback.log_dir, trial.trial_id, execution
                )
                callback.log_dir = logdir
                hparams = keras_tuner.engine.tuner_utils.convert_hyperparams_to_hparams(
                    trial.hyperparameters
                )
                callbacks.append(
                    hparams_api.KerasCallback(
                        writer=logdir, hparams=hparams, trial_id=trial.trial_id
                    )
                )

    def _get_tensorboard_dir(self, logdir, trial_id, execution):
        return os.path.join(str(logdir), str(trial_id), f"execution{str(execution)}")

    def _get_checkpoint_fname(self, trial_id, execution):
        return os.path.join(
            # Each checkpoint is saved in its own directory.
            self.get_trial_dir(trial_id),
            "ckpt_exec" + str(execution) + ".h5"
        )


class Logger(Callback):

    def on_train_begin(self, logs=None):
        # Create scores holder
        global val_score_holder
        val_score_holder = []
        global test_score_holder
        test_score_holder = []
        global train_score_holder
        train_score_holder = []

    def on_epoch_end(epoch, logs):
        # Access tuner and logger from the global workspace
        global val_score_holder
        global train_score_holder
        # global test_score_holder
        # print(logs)
        # Store scores
        val_score_holder.append(logs['val_loss'])
        # test_score_holder.append(logs['test_loss'])
        train_score_holder.append(logs['loss'])

    def on_train_end(logs=None):
        # Access tuner and score holders from the global workspace
        global tuner
        global val_score_holder
        global train_score_holder
        global test_score_holder

        # Get last (current) trial
        trial = list(tuner.oracle.trials.keys())[-1]

        # Create new attributes if not already present i.e. new trial
        if 'rep_val_loss' not in dir(tuner.oracle.trials[trial]):
            tuner.oracle.trials[trial].rep_val_loss = []
            tuner.oracle.trials[trial].rep_test_loss = []
            tuner.oracle.trials[trial].rep_train_loss = []

        # Add min val loss and corresponding training loss
        tuner.oracle.trials[trial].rep_val_loss.append(np.min(val_score_holder))
        tuner.oracle.trials[trial].rep_train_loss.append(train_score_holder[np.argmin(val_score_holder)])
        tuner.oracle.trials[trial].rep_test_loss.append(logs['best_test_loss'])


def imprimeTabela(path, tuner):
    print(tuner.oracle.max_trials)
    min_res = np.zeros(tuner.oracle.max_trials)
    mean_res = np.zeros(tuner.oracle.max_trials)
    var_res = np.zeros(tuner.oracle.max_trials)
    exec_best = np.zeros(tuner.oracle.max_trials)

    title = path + '/autokeras_mlp4tcn_res_exp2_all/NAS_results.csv'
    results = open(title, "w+")

    # Header

    results.write('"Trial_id",')
    results.write('"Best exec",')
    results.write('"Min test MSE",')
    results.write('"Mean test MSE",')
    results.write('"Variance test MSE",')
    results.write('"Hidden",')
    results.write('"Nb_filters",')
    results.write('"k_size",')
    results.write('"dense",')
    results.write('"params"')
    results.write('\n')

    for n in range(tuner.oracle.max_trials):
        if tuner.oracle.max_trials >= 10:
            trial_id = "{:02d}".format(n)
        else:
            trial_id = "{:01d}".format(n)
        # print("trial_id:",trial_id)
        # print(tuner.oracle.trials['0'])
        min_res[n] = (stats.describe(np.array(tuner.oracle.trials[trial_id].rep_test_loss)))[1][0]
        mean_res[n] = (stats.describe(np.array(tuner.oracle.trials[trial_id].rep_test_loss)))[2]
        var_res[n] = (stats.describe(np.array(tuner.oracle.trials[trial_id].rep_test_loss)))[3]
        exec_best[n] = np.argmin(tuner.oracle.trials[trial_id].rep_test_loss)
        hp_tmp = tuner.oracle.trials[trial_id].hyperparameters

        model_tmp = tuner.hypermodel.build(hp_tmp)
        params = model_tmp.count_params()

        results.write('%d,' % n)
        results.write('%d,' % exec_best[n])
        results.write('%.5f,' % min_res[n])
        results.write('%.5f,' % mean_res[n])
        results.write('%.5f,' % var_res[n])
        results.write('%d,' % hp_tmp['hidden'])
        results.write('%d,' % hp_tmp['nb_filters'])
        results.write('%d,' % hp_tmp['k_size'])
        results.write('%d,' % hp_tmp['dense'])
        results.write('%d' % params)

        results.write('\n')
    results.close()

    best_test_loss_trial = np.argmin(min_res)
    best_test_loss_exec = exec_best[best_test_loss_trial]

    return best_test_loss_trial, best_test_loss_exec


# Tuner instatiation and search
# tuner = BayesianOptimization(

tuner = GridSearchTuner(
    MyHyperModel(),
    objective=keras_tuner.Objective("test_loss", "min"),
    max_trials=80,
    executions_per_trial=10,
    directory=path,
    project_name="autokeras_mlp4tcn_res_exp2_all",
    max_model_size=300000,
    overwrite=True)

logger_callback = Logger
# start search
tuner.search_space_summary(extended=True)
tuner.search(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, path_save=path, callbacks=[logger_callback])

# save search res and load best model
trial_id_tmp, exec_id = imprimeTabela(path, tuner)
if tuner.oracle.max_trials >= 10:
    trial_id = "{:02d}".format(trial_id_tmp)
else:
    trial_id = "{:01d}".format(trial_id_tmp)

title = path + '/bestNAS_results.csv'
results = open(title, "w+")
results.write('"Trial_id",')
results.write('"Best exec",')
results.write('"Min test MSE",')
results.write('"Mean test MSE",')
results.write('"Variance test MSE",')
results.write('"Hidden",')
results.write('"Nb_filters",')
results.write('"k_size",')
results.write('"dense",')
results.write('"params"')
results.write('\n')

min_res_best = (stats.describe(np.array(tuner.oracle.trials[trial_id].rep_test_loss)))[1][0]
mean_res_best = (stats.describe(np.array(tuner.oracle.trials[trial_id].rep_test_loss)))[2]
var_res_best = (stats.describe(np.array(tuner.oracle.trials[trial_id].rep_test_loss)))[3]
exec_best_best = np.argmin(tuner.oracle.trials[trial_id].rep_test_loss)
hp_tmp = tuner.oracle.trials[trial_id].hyperparameters

model_tmp = tuner.hypermodel.build(hp_tmp)
params = model_tmp.count_params()

results.write('%s,' % trial_id)
results.write('%d,' % exec_best_best)
results.write('%.5f,' % min_res_best)
results.write('%.5f,' % mean_res_best)
results.write('%.5f,' % var_res_best)
results.write('%d,' % hp_tmp['hidden'])
results.write('%d,' % hp_tmp['nb_filters'])
results.write('%d,' % hp_tmp['k_size'])
results.write('%d,' % hp_tmp['dense'])
results.write('%d' % params)

results.write('\n')