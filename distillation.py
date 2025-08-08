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
import tensorflow as tf
import tcn

from keras import regularizers, optimizers, losses, initializers, metrics
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model, model_from_json
from keras import layers, models
from keras import backend as K
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tcn import TCN, tcn_full_summary, compiled_tcn

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    
batch_size = 16
input_sequence_length = 15
output_sequence_length = 1
multi_horizon = False
window_time = 5
window_size = 15
epochs=2000
lr=0.0001
alpha=0.1

parent_dir =  "/home/giorgia/TCNDistilledbest_distillAIL_EXP1/"
directory = 'TCNBestloss'
path = os.path.join(parent_dir, directory)
if not os.path.exists(path):
    os.mkdir(path)
    print("Directory '%s' created" %directory)


hidden_stud =3
nb_stack_stud=1
nb_filters_stud = 8
kernel_size_stud=5
dense_unit_stud=8


def TCNmodel(nb_filters,num_units,kernel_size,hidden,nb_stacks,input_shape):   
    x = layers.Input(shape=input_shape)
    tcn_out = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=1, dilations=[2 ** i for i in range(hidden)], padding = 'same', use_skip_connections='True', dropout_rate=0.05, return_sequences=False ,activation='relu', kernel_initializer='glorot_uniform', use_layer_norm= True, name='tcn')(x)
    flatten_out = layers.Flatten()(tcn_out)
    dense_out1 = layers.Dense(num_units, activation='linear')(flatten_out)
    dense_out = layers.Dense(2)(dense_out1)
    #dense_out = layers.Reshape((1,2))(dense_out)
    saved_model = models.Model(x, dense_out)
        
    return saved_model

#Load Dataset
def load_dataset(path):
    training_data = "preprocessed-training-Osama.csv"
    validation_data = "preprocessed-validation-Osama.csv"
    testing_data = "preprocessed-testing-Osama.csv"
    trainList = list()
    valList = list()
    testList = list()

    with open(training_data, 'r') as train_inp_csv:
        train_inp_csv_reader = csv.reader(train_inp_csv)
        for rowTr in train_inp_csv_reader:
            trainList.append(rowTr)

    trainArray = np.asarray(trainList, dtype=np.float32)
    #print(trainArray.shape)
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

    # plt.figure()
    # plt.plot(Y_train[:,0],Y_train[:,1])
    # plt.plot(Y_val[:,0],Y_val[:,1])
    # # plt.plot(Y_test[:,0],Y_test[:,1])
    # plt.legend(['training','validation'])

    len_train = X_train.shape[0]
    len_val = X_val.shape[0]
    len_test = X_test.shape[0]

    print(len_train, len_val)   


    plt.plot(Y_train[:,0],Y_train[:,1],'cornflowerblue', label="training")
    plt.plot(Y_val[:,0],Y_val[:,1],'red', label="validation")
    plt.plot(Y_test[:,0],Y_test[:,1],'green', label="testing")
    plt.grid()
    plt.xlim(0,3)
    plt.ylim(0,3)
    #plt.legend(['training','validation',"test"])
    plt.legend(loc='upper right')
    plt.savefig(path +'/Paths_reduced.pdf',bbox_inches='tight')
    plt.close()
    
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def create_tf_dataset(
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

    target_offset = math.floor(input_sequence_length/2)
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

def train(teacher_model, student_model, lr, path_save,epochs, n, train_ds,val_ds, alpha):
    loss_values = np.zeros(epochs)
    val_loss_values = np.zeros(epochs)
    best_epoch=0
    best_val_loss = 100
    # Average the loss across the batch size within an epoch
    student_loss_fn =losses.MSE

    # Specify the performance metric
    optimizer = tf.keras.optimizers.Adamax(lr=lr)
    train_acc = tf.keras.metrics.MeanSquaredError()
    valid_acc = tf.keras.metrics.MeanSquaredError()
    
    alpha=alpha
    #eta must be calculated from max min teacher errors
    eta=1.181509971618650

    @tf.function(jit_compile=True)
    def train_step(x,y,teacher_model,student_model):
        # Unpack data

        # Forward pass of teacher
        teacher_predictions = teacher_model(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = student_model(x, training=True)

            # Compute losses
            distillation_loss=student_loss_fn(teacher_predictions, student_predictions)
            student_loss = student_loss_fn(y, student_predictions)
            teacher_loss = student_loss_fn(y, teacher_predictions)

            loss = alpha * student_loss + (1 - alpha) * (1-(teacher_loss/eta))*distillation_loss
            

        # Compute gradients
        trainable_vars = student_model.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        train_acc.update_state(y, student_predictions)
        #print("mse:",self.compiled_metrics.result())

        return distillation_loss, loss
    @tf.function(jit_compile=True)
    def test_step(x,y, teacher_model,student_model):
        # Compute predictions
        y_prediction = student_model(x, training=False)
        # Calculate the loss
        val_student_loss = student_loss_fn(y, y_prediction)
        # Update the metrics.
        valid_acc.update_state(y, y_prediction)
        #print("val_loss:",  self.compiled_metrics.result())
        # Return a dict of performance
        return val_student_loss
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for step, (X_train, Y_train) in enumerate(train_ds):
            distillation_loss,loss=train_step(X_train, Y_train, teacher_model, student_model)
            #print(step)
        loss_values[epoch] = train_acc.result()
        #print("Training loss epoch %d: %.4f" %  (epoch,float(loss_values[epoch])))

        # Reset training metrics at the end of each epoch
        train_acc.reset_states()
        
        for step_val, (X_val, Y_val) in enumerate(val_ds):
            test_step(X_val, Y_val, teacher_model, student_model)
            #print(step_val)
            
        val_loss_values[epoch] = valid_acc.result()
        valid_acc.reset_states()
        #print("Validation loss epoch %d: %.4f" %(epoch, float(val_loss_values[epoch])))
        
        template = "Epoch {}, train_loss: {:.3f}, val_loss: {:.3f}"
        print (template.format(epoch+1,
                            loss_values[epoch],
                            val_loss_values[epoch]))
        if epoch == 0:
            wait=0
            student_model.save_weights(path_save+'/best_model'+str(n)+'.h5')
            best_epoch = epoch
            best_val_loss= val_loss_values[epoch]
        if val_loss_values[epoch] < best_val_loss and epoch > 0:
            wait=0
            student_model.save_weights(path_save+'/best_model'+str(n)+'.h5')
            print("Validation loss improved, model weights are saved" )
            best_epoch = epoch
            best_val_loss= val_loss_values[epoch]
        if val_loss_values[epoch] >= best_val_loss and epoch > 10:
            wait=wait+1
        if wait == 80:
            break
    
    epochs_num = range(1, len(loss_values) + 1)
           
    plt.figure()
    #
    # Plot the model accuracy vs Epochs
    #
    plt.plot(epochs_num[0:epoch], loss_values[0:epoch], 'r', label='Training loss')
    plt.plot(epochs_num[0:epoch], val_loss_values[0:epoch], 'b', label='Validation loss')
    plt.suptitle('Training & Validation Loss', fontsize=16)
    plt.title("Learning Curves", fontsize=12)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.savefig(path_save +'/Training&Validation'+str(n)+'.pdf')
    plt.close()
        
    
    return teacher_model, student_model

def test(saved_model, train_dataset, valid_dataset, test_dataset, path, batch_size, epochs, n, window_size, kernel_size):
    
    trnable_params = saved_model.count_params()
    sliced_train_X, sliced_train_Y = tuple(zip(*train_dataset))
    sliced_train_X = np.array(sliced_train_X)[:,0,:,:]
    sliced_train_Y = np.array(sliced_train_Y)[:,0,0,:]

    sliced_val_X,sliced_val_Y = tuple(zip(*valid_dataset))
    sliced_val_X = np.array(sliced_val_X)[:,0,:,:]
    sliced_val_Y = np.array(sliced_val_Y)[:,0,0,:]

    sliced_test_X,sliced_test_Y = tuple(zip(*test_dataset))
    sliced_test_X = np.array(sliced_test_X)[:,0,:,:]
    sliced_test_Y = np.array(sliced_test_Y)[:,0,0,:]

    preds_train = (saved_model(sliced_train_X, training=False))[:,0,:]
    #print("shape preds:", preds_train.shape)
    #print("shape Y", sliced_test_Y.shape)
    preds_val = (saved_model(sliced_val_X, training=False))[:,0,:]
    preds_test = (saved_model(sliced_test_X, training=False))[:,0,:]
    
    
    eval_test = saved_model.evaluate(sliced_test_X, sliced_test_Y, batch_size = None , verbose=1)
    # print("test: ", eval_test)
    # print(saved_model.metrics_names)

    eval_val = saved_model.evaluate(sliced_val_X, sliced_val_Y, batch_size = None , verbose=1)
    # print("validation: ", eval_val)
    # print(saved_model.metrics_names)

    eval_train = saved_model.evaluate(sliced_train_X, sliced_train_Y, batch_size = None , verbose=1)
    # print("training: ", eval_train)
    # print(saved_model.metrics_names)
    
    mse_test=mean_squared_error(sliced_test_Y[:,:],preds_test[:,:])
    mse_train=mean_squared_error(sliced_train_Y[:,:],preds_train[:,:])
    mse_val=mean_squared_error(sliced_val_Y[:,:],preds_val[:,:])

    plt.figure()
    plt.plot(sliced_train_Y[:,0],sliced_train_Y[:,1])
    plt.plot(preds_train[:,0],preds_train[:,1])
    plt.legend(['true','pred'])
    plt.title('Training')
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.savefig(path +'/Training_pred'+str(n)+'.pdf')
    plt.close()


    plt.figure()
    plt.plot(sliced_val_Y[:,0],sliced_val_Y[:,1])
    plt.plot(preds_val[:,0],preds_val[:,1])
    plt.legend(['true','pred'])
    plt.title('Validation')
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.savefig(path +'/Validation_pred'+str(n)+'.pdf')
    plt.close()

    plt.figure()
    plt.plot(sliced_test_Y[:,0],sliced_test_Y[:,1])
    plt.plot(preds_test[:,0],preds_test[:,1])
    plt.legend(['true','pred'])
    plt.title('Test')
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.savefig(path +'/Test_pred'+str(n)+'.pdf')
    plt.close()
    
      ## Plot results
    ##

    testcomXplot = plt.figure(1)
    plt.plot(preds_test[:,0], '-b', label = 'prediction')
    plt.plot(sliced_test_Y[:,0], 'r', label = 'reference')
    plt.legend(loc='upper left')
    plt.title("Ground truth vs prediction for X axis - test set")
    #plt.xlim(0,3)
    plt.ylim(0,3)
    plt.grid()
    plt.savefig(path+'/nn-test-trajectory-x-ground-truth-and-prediction'+str(n)+'.pdf', bbox_inches='tight')
    #testcomXplot.show()
    plt.close()


    testcomYplot = plt.figure(2)
    plt.plot(preds_test[:,1], '-b', label = 'prediction')
    plt.plot(sliced_test_Y[:,1], 'r', label = 'reference')
    plt.legend(loc='upper left')
    plt.title("Ground truth vs prediction for Y axis - test set")
    #plt.xlim(0,3)
    plt.ylim(0,3)
    plt.grid()
    plt.savefig(path+'/nn-test-trajectory-y-ground-truth-and-prediction'+str(n)+'.pdf', bbox_inches='tight')
    #testcomYplot.show()
    plt.close()

    valcomXplot = plt.figure(1)
    plt.plot(preds_val[:,0], '-b', label = 'prediction')
    plt.plot(sliced_val_Y[:,0], 'r', label = 'reference')
    plt.legend(loc='upper left')
    plt.title("Ground truth vs prediction for X axis - validation set")
    #plt.xlim(0,3)
    plt.ylim(0,3)
    plt.grid()
    plt.savefig(path+'/nn-validation-trajectory-x-ground-truth-and-prediction'+str(n)+'.pdf', bbox_inches='tight')
    plt.close()
    #testcomXplot.show()

    valcomYplot = plt.figure(2)
    plt.plot(preds_val[:,1], '-b', label = 'prediction')
    plt.plot(sliced_val_Y[:,1], 'r', label = 'reference')
    plt.title("Ground truth vs prediction for Y axis - validation set")
    plt.legend(loc='upper left')
    #plt.xlim(0,3)
    plt.ylim(0,3)
    plt.grid()
    plt.savefig(path+'/nn-validation-trajectory-y-ground-truth-and-prediction'+str(n)+'.pdf', bbox_inches='tight')
    #testcomYplot.show()

    plt.close('all')



    #### Here I calculate the Euclidean distance ####

    dist_euc_train = np.mean(np.sqrt(np.sum(np.square(preds_train - sliced_train_Y), axis=-1)))
    dist_euc_dev = np.mean(np.sqrt(np.sum(np.square(preds_val - sliced_val_Y), axis=-1)))
    dist_euc_test = np.mean(np.sqrt(np.sum(np.square(preds_test - sliced_test_Y), axis=-1)))

    print("Train Euclidean distance: " + str(dist_euc_train) )
    print("Val Euclidean distance: " + str(dist_euc_dev) )
    print("Test Euclidean distance: " + str(dist_euc_test) )
    
    #write to the file following numbers
    #number of epochs
    #training rate
    #batch size
    #Accuracy/training/tesing/val
    #MSE training/testing/val

    title='/nn-results'+str(n)+'.csv'
    results = open(path+title, "w+")

    #Basic pre-reqs:

    results.write('"Data slicing window(smpl)",')
    results.write('"Epochs",')
    results.write('"Training MSE",')
    results.write('"Validation MSE",')
    results.write('"Test MSE",')

    results.write('"Training Euclidean",')
    results.write('"Validation Euclidean",')
    results.write('"Test Euclidean",')

    results.write('"Number of parameters"')
    # results.write('"kernel size",')

    results.write('\n')

    results.write('%d,' % window_size)
    results.write('%d,' % epochs)
    results.write('%f,' % mse_train)
    results.write('%f,' % mse_val)
    results.write('%f,' % mse_test)


    results.write('%f,' % dist_euc_train)
    results.write('%f,' % dist_euc_dev)
    results.write('%f,' % dist_euc_test)

    results.write('%f' % trnable_params)
    # results.write('%f,' % kernel_size)

    results.write('\n')

    results.close()

    return preds_test


#Load and create Datasets
X_train, Y_train, X_val, Y_val, X_test, Y_test= load_dataset(parent_dir)
train_dataset = create_tf_dataset(X_train, Y_train, input_sequence_length, output_sequence_length, batch_size)
val_dataset = create_tf_dataset(X_val, Y_val, input_sequence_length, output_sequence_length, batch_size)
test_dataset = create_tf_dataset(X_test, Y_test, input_sequence_length, output_sequence_length, batch_size)

train_dataset_res = create_tf_dataset(X_train, Y_train, input_sequence_length, output_sequence_length, 1)
val_dataset_res = create_tf_dataset(X_val, Y_val, input_sequence_length, output_sequence_length, 1)
test_dataset_res = create_tf_dataset(X_test, Y_test, input_sequence_length, output_sequence_length, 1)


#Prepare directories and Train&Validate
parent_dir_save =  "/home/giorgia/TCNDistilledbest_distillAIL_EXP1/"
if not os.path.exists(parent_dir_save):
    os.mkdir(parent_dir_save)

directory_save = 'TCNresult_'+str(lr)
path_save = os.path.join(parent_dir_save, directory_save)
if not os.path.exists(path_save):
    os.mkdir(path_save)
    print("Directory '%s' created" %directory_save)

for i in range (0,20):
    K.clear_session()
    student_model= TCNmodel(nb_filters_stud, dense_unit_stud, kernel_size_stud, hidden_stud, nb_stack_stud,[15,4])
    student_params = student_model.count_params()
    print(student_params)
    student_model.load_weights(path+'/best_model0.h5')
    print("weights loaded")
    student_model_final= train(student_model, lr, path_save, epochs,i,train_dataset,val_dataset, alpha)

    model_as_json = student_model_final.to_json()
    with open(path_save+'/model'+str(i)+'.json', "w") as json_file:
        json_file.write(model_as_json)
        
    student_model_final.load_weights(path_save+'/best_model'+str(i)+'.h5')
    student_model_final.compile('adam','mse')
    test(student_model_final, train_dataset_res, val_dataset_res, test_dataset_res, path_save, batch_size, epochs,i, window_size, kernel_size_stud)