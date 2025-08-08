import os
import tensorflow as tf
from sklearn.metrics import classification_report
#import autokeras as ak
import keras_tuner
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
import keras
from tensorboard.plugins.hparams import api as hparams_api
import copy
from keras import regularizers, optimizers, losses, initializers, metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import backend as K
from keras import backend as K
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from scipy import stats

# if lbs=="Cap":
#     freq=3
#     channel=4
#     input_sequence_length = 15
#     output_sequence_length = 1
#     window_size = 15
#     window_time = 5
#     folder="cap"
# elif lbs=="Radar":
#     freq=4
#     channel=3
#     input_sequence_length = 13
#     output_sequence_length = 1
#     window_size = 13
#     window_time = 3
#     folder="radar"
# elif lbs=="Ir":
#     freq=5
#     channel=16
#     input_sequence_length = 5
#     output_sequence_length = 1
#     window_size = 5
#     window_time = 1
#     folder="ir"


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_dataset_cap(path,file_training,file_valid,file_testing):
    training_data =file_training
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
    #print(trainArray.shape)
    # x > 0.70712,0.88478,0.93737,0.82217,y > 2.314,2.342

    train_X = trainArray[:, 0:4]
    train_Y = trainArray[:, 4:6]
    #####################################

    with open(validation_data, 'r') as val_inp_csv:
        val_inp_csv_reader = csv.reader(val_inp_csv)
        for rowVl in val_inp_csv_reader:
            valList.append(rowVl)

    valArray = np.asarray(valList, dtype=np.float32)
    #print(valArray.shape)
    
    val_X = valArray[:, 0:4]
    val_Y = valArray[:, 4:6]
    #####################################

    with open(testing_data, 'r') as test_inp_csv:
        test_inp_csv_reader = csv.reader(test_inp_csv)
        for rowTe in test_inp_csv_reader:
            testList.append(rowTe)

    testArray = np.asarray(testList, dtype=np.float32)
    #print(valArray.shape)
    
    test_X = testArray[:, 0:4]
    test_Y = testArray[:, 4:6]


    X_train = np.array(train_X)   #np.transpose(train_X)
    Y_train = np.array(train_Y)   #np.transpose(train_Y)
 

    X_val = np.array(val_X)   #np.transpose(test1_X)
    Y_val = np.array(val_Y)   #np.transpose(test1_Y)

    
    X_test = np.array(test_X)   #np.transpose(test2_X)
    Y_test = np.array(test_Y)   #np.transpose(test2_Y)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_dataset_ir(path,file_training,file_valid,file_testing):
    training_data =file_training
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
    #print(trainArray.shape)
    train_X = trainArray[:, 0:64]
    train_Y = trainArray[:, 64:]
    #####################################

    with open(validation_data, 'r') as val_inp_csv:
        val_inp_csv_reader = csv.reader(val_inp_csv)
        for rowVl in val_inp_csv_reader:
            valList.append(rowVl)

    valArray = np.asarray(valList, dtype=np.float32)
    #print(valArray.shape)
    
    val_X = valArray[:, 0:64]
    val_Y = valArray[:, 64:]
    #####################################

    with open(testing_data, 'r') as test_inp_csv:
        test_inp_csv_reader = csv.reader(test_inp_csv)
        for rowTe in test_inp_csv_reader:
            testList.append(rowTe)

    testArray = np.asarray(testList, dtype=np.float32)
    #print(valArray.shape)
    
    test_X = testArray[:, 0:64]
    test_Y = testArray[:, 64:]


    X_train = np.array(train_X)   #np.transpose(train_X)
    Y_train = np.array(train_Y)   #np.transpose(train_Y)
 

    X_val = np.array(val_X)   #np.transpose(test1_X)
    Y_val = np.array(val_Y)   #np.transpose(test1_Y)

    
    X_test = np.array(test_X)   #np.transpose(test2_X)
    Y_test = np.array(test_Y)   #np.transpose(test2_Y)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_dataset_radar(path,file_training,file_valid,file_testing):
    training_data =file_training
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
    #print(trainArray.shape)
    train_X = trainArray[:, 0:3]
    train_Y = trainArray[:, 3:5]
    #####################################

    with open(validation_data, 'r') as val_inp_csv:
        val_inp_csv_reader = csv.reader(val_inp_csv)
        for rowVl in val_inp_csv_reader:
            valList.append(rowVl)

    valArray = np.asarray(valList, dtype=np.float32)
    #print(valArray.shape)
    
    val_X = valArray[:, 0:3]
    val_Y = valArray[:, 3:5]
    #####################################

    with open(testing_data, 'r') as test_inp_csv:
        test_inp_csv_reader = csv.reader(test_inp_csv)
        for rowTe in test_inp_csv_reader:
            testList.append(rowTe)

    testArray = np.asarray(testList, dtype=np.float32)
    #print(valArray.shape)
    
    test_X = testArray[:, 0:3]
    test_Y = testArray[:, 3:5]


    X_train = np.array(train_X)   #np.transpose(train_X)
    Y_train = np.array(train_Y)   #np.transpose(train_Y)
 

    X_val = np.array(val_X)   #np.transpose(test1_X)
    Y_val = np.array(val_Y)   #np.transpose(test1_Y)

    
    X_test = np.array(test_X)   #np.transpose(test2_X)
    Y_test = np.array(test_Y)   #np.transpose(test2_Y)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def create_tf_dataset_cap(
    data_array: np.ndarray,
    output_array: np.ndarray, 
    input_sequence_length: int,
    output_sequence_length: int,
    batch_size: int = 1,
    shuffle=False,
    multi_horizon=False,
):
    """Creates tensorflow dataset from numpy array.

    This function creates a dataset where each element is a tuple `(inputs, targets)`.
    `inputs` is a Tensor
    of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
    the `input_sequence_length` past values of the timeseries for each node.
    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
    containing the `forecast_horizon`
    future values of the timeseries for each node.

    Args:
        data_array: np.ndarray with shape `(num_time_steps, num_routes)`
        input_sequence_length: Length of the input sequence (in number of timesteps).
        forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
            `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
            timeseries `forecast_horizon` steps ahead (only one value).
        batch_size: Number of timeseries samples in each batch.
        shuffle: Whether to shuffle output samples, or instead draw them in chronological order.
        multi_horizon: See `forecast_horizon`.

    Returns:
        A tf.data.Dataset instance.
    """

    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-2,:], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = math.floor(input_sequence_length/2)+1
    target_seq_length = output_sequence_length
    targets = timeseries_dataset_from_array(
        output_array[target_offset:-target_offset,:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset

def compute_sparc(x_positions,y_positions,fs,fc_max,padlevel=0):

    d_x = np.diff(x_positions)
    d_y = np.diff(y_positions)
    
    dx = np.diff(x_positions)*fs
    dy = np.diff(y_positions)*fs
    
   
    
    ddx = np.diff(dx)*fs
    ddy = np.diff(dy)*fs
    
    dddx = np.diff(ddx)*fs
    dddy = np.diff(ddy)*fs
    
    movement_pos=np.sqrt(d_x**2 + d_y**2)
    #movement_vel=np.sqrt(dx**2 + dy**2)
    # movement_vel = (movement_pos)*fs
    
    movement_vel1 = (movement_pos)*fs
    movement_vel = np.diff(movement_vel1)*fs
    
    # movement_pos= np.sqrt(x_positions**2 + y_positions**2)
    # movement_vel= np.diff(movement_pos)*fs
    
    #movement_t=movement_vel
    
    window = np.hanning(len(movement_vel))
    movement_t = (movement_vel - np.mean(movement_vel)) * window
    
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement_t))+padlevel)))
    movement_new= np.pad(movement_t,(0,nfft-len(movement_t)),'constant')
    
    print(nfft)
    print(len(movement_new))
    movement=movement_new
    # Frequency
    f = np.fft.fftfreq(movement.shape[0], 1/fs)

    Mf = abs(np.fft.fft(movement,nfft))
    Mf = Mf / max(Mf)

    fc_inx = np.nonzero((f >= 0) & (f <= fc_max ))
    
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]
    
    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    #inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    #fc_inx = range(inx[0], inx[-1] + 1)
    
    f_sel = f_sel[fc_inx[:-1]]
    Mf_sel = Mf_sel[fc_inx[:-1]]
    
    # Calculate arc length
    new_sal = sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)

def compute_test(saved_model,n,X_train, Y_train,X_val, Y_val,X_test, Y_test,path,trnable_params,nb_filters,kernel_size,hidden,nb_stacks,dense):
    eval_test = saved_model.evaluate(X_test, Y_test, batch_size = None , verbose=1)
    print("test: ", eval_test)


    eval_val = saved_model.evaluate(X_val, Y_val, batch_size = None , verbose=1)
    print("validation: ", eval_val)


    eval_train = saved_model.evaluate(X_train, Y_train, batch_size = None , verbose=1)
    print("training: ", eval_train)


    preds_train_tcn = (saved_model.predict(X_train, batch_size = None))
    preds_val_tcn = (saved_model.predict(X_val, batch_size = None))
    preds_test_tcn = (saved_model.predict(X_test, batch_size = None))


        

    plt.figure()
    plt.plot(Y_train[:,0],Y_train[:,1])
    plt.plot(preds_train_tcn[:,0],preds_train_tcn[:,1])
    plt.legend(['true', 'pred'])
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)
    plt.savefig(path + '/Training.pdf')
    plt.close()



    plt.figure()
    plt.plot(Y_val[:,0],Y_val[:,1])
    plt.plot(preds_val_tcn[:,0],preds_val_tcn[:,1])
    plt.legend(['true', 'pred'])
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)
    plt.savefig(path + '/Validation.pdf')
    plt.close()

    plt.figure()
    plt.plot(Y_test[:,0],Y_test[:,1])
    plt.plot(preds_test_tcn[:,0],preds_test_tcn[:,1])
    plt.legend(['true','pred'])
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)
    plt.savefig(path + '/Testing.pdf')
    plt.close()


    mse_test=mean_squared_error(Y_test[:,:],preds_test_tcn[:,:])
    mse_train=mean_squared_error(Y_train[:,:],preds_train_tcn[:,:])
    mse_val=mean_squared_error(Y_val[:,:],preds_val_tcn[:,:])

        
    dist_euc_train = np.mean(np.sqrt(np.sum(np.square(preds_train_tcn[:,:] - Y_train[:,:]), axis=-1)))
    dist_euc_dev = np.mean(np.sqrt(np.sum(np.square(preds_val_tcn[:,:] - Y_val[:,:]), axis=-1)))
    dist_euc_test = np.mean(np.sqrt(np.sum(np.square(preds_test_tcn[:,:] - Y_test[:,:]), axis=-1)))


    output_csv = np.concatenate([preds_train_tcn, Y_train], axis = 1)
    csv_out_name = path + "/results_train_vs_actual.csv"
    with open(csv_out_name, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(output_csv)
    
    output_csv2 = np.concatenate([preds_val_tcn, Y_val], axis=1)
    csv_out_name2 = path+"/results_val_vs_actual.csv"
    with open(csv_out_name2, "w") as output2:
        writer = csv.writer(output2, lineterminator='\n')
        writer.writerows(output_csv2)  
        
    output_csv3 = np.concatenate([preds_test_tcn, Y_test], axis=1)
    csv_out_name3 = path+"/results_test_vs_actual.csv"
    with open(csv_out_name3, "w") as output3:
        writer = csv.writer(output3, lineterminator='\n')
        writer.writerows(output_csv3)  
        
        
    #Write results:
    title='/nn-result-test.csv'
    results = open(path+title, "w+")

    results.write('"Training MSE",')
    results.write('"Validation MSE",')
    results.write('"Testing EXP3 MSE",')

    results.write('"Training Euclidean",')
    results.write('"Validation Euclidean",')
    results.write('"Testing Euclidean",')

    results.write('"Number of parameters",')
    results.write('"N filters",')
    results.write('"kernel size",')
    results.write('"N stacks",')
    results.write('"Hidden",')
    results.write('"Dense unit"')

    results.write('\n')

    results.write('%f,' % mse_train)
    results.write('%f,' % mse_val)
    results.write('%f,' % mse_test)


    results.write('%f,' % dist_euc_train)
    results.write('%f,' % dist_euc_dev)
    results.write('%f,' % dist_euc_test)

    results.write('%f,' % trnable_params)
    results.write('%f,' % nb_filters)
    results.write('%f,' % kernel_size)
    results.write('%f,' % nb_stacks)
    results.write('%f,' % hidden)
    results.write('%f' % dense)

    results.write('\n')

    results.close()