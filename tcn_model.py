import tensorflow as tf
import keras
import numpy as np
from ncps import wirings
from ncps.tf import LTC, CfC
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model, model_from_json
from keras import layers, models
from tensorflow.keras.layers import Input, TimeDistributed, Concatenate, Lambda
from keras import backend as K
from keras.callbacks import Callback
from keras import regularizers, optimizers, losses, initializers, metrics
from tcn_simple import TCN_model
from tcn_old import TCN
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

def model_TCN_simple(hidden,num_filters,k_size,dense, name='tcn'):
    #x = layers.Input(shape=(15, 4), name= f"{name}_input")
    x = keras.Input(shape=(20, 64), name=f"{name}_input")
    model = TCN_model(num_layers=hidden,
                    num_filters=num_filters,
                    kernel_size=k_size,
                    dilations=[2 ** i for i in range(hidden)],
                    output_len=1)
    print("tcn block configure:", model.get_config())
    tcn_out = model(x)
    # tcn_out = TCN_model(num_layers=hidden,
    #                 num_filters=num_filters,
    #                 kernel_size=k_size,
    #                 dilations=[2 ** i for i in range(hidden)],
    #                 output_len=1)(x)
    print("shape_out_tcn:", tcn_out.shape)
    dense_out1 = layers.Dense(dense, name=f"{name}_dense_out_1")(tcn_out)
    dense_out = layers.Dense(2, name=f"{name}_dense_out_2")(dense_out1)
    print("dense_out:", dense_out.shape)
    model = models.Model(x, dense_out, name="tcn_model")
    return model

def model_TCN_complete(hidden,num_filters,k_size,dense, name='tcn'):
    x = layers.Input(shape=(20, 64),name= f"{name}_input")
    tcn_out = TCN(nb_filters=num_filters, kernel_size=k_size, nb_stacks=1, dilations=[2 ** i for i in range(hidden)], padding = 'same', use_skip_connections='True', dropout_rate=0.01, return_sequences=False ,activation='relu', kernel_initializer='glorot_uniform', use_layer_norm= True, name='tcn')(x)
    flatten_out = layers.Flatten()(tcn_out)
    dense_out1=layers.Dense(dense,name= f"{name}_dense_out_1")(flatten_out)
    dense_out = layers.Dense(2, name= f"{name}_dense_out_2")(dense_out1)
    print("dense_out:", dense_out.shape)
    model = models.Model(x, dense_out,name="tcn_model")
    return model



def LTCModel(input_shape,unit):  
    wiring = wirings.FullyConnected(unit) 
    x = layers.Input(shape=input_shape)
    _,ltc_out_right=  (LTC(wiring,return_sequences=False,return_state=True, ode_unfolds=10))(x)
    _,ltc_out_left=  (LTC(wiring,return_sequences=False,go_backwards=True,return_state=True,ode_unfolds=10))(x)
    concatenate=layers.concatenate([ltc_out_right,ltc_out_left])
    dense_out= layers.Dense(2)(concatenate)
    saved_model = models.Model(x, dense_out) 
    return saved_model
def NCPModel(input_shape,unit):  
    wiring = wirings.AutoNCP(units=unit,output_size=4,sparsity_level=0.5) 
    x = layers.Input(shape=input_shape)
    #1st out, 2nd state
    _,ltc_out_right=  (LTC(wiring,return_sequences=False,return_state=True, ode_unfolds=10))(x)
    _,ltc_out_left=  (LTC(wiring,return_sequences=False,go_backwards=True,return_state=True,ode_unfolds=10))(x)
    concatenate=layers.concatenate([ltc_out_right,ltc_out_left])
    dense_out= layers.Dense(2)(concatenate)
    saved_model = models.Model(x, dense_out) 
    return saved_model

def CPSmodel_cap(input_shape, n_class, routings, nb_filters, kernel_size, dim_capsule_caps1, num_capsule_caps1, dim_capsule_caps2, dense_unit):   

    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv2')(conv1)
    #from[20,4] to [18,256]
    print("Output 1st Conv: ", conv1.shape)
    
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=dim_capsule_caps1, n_channels=num_capsule_caps1, kernel_size=kernel_size, strides=2, padding='valid')
    
    print("primarycaps_out: ",primarycaps.shape)
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule_caps2, num_routing=routings,
                             name='digitcaps')(primarycaps)
   
    
    digitcaps = K.squeeze(digitcaps, axis=1)
    digitcaps = K.squeeze(digitcaps, axis=3)
    #digitcaps = layers.Reshape((n_class, dim_capsule_caps2))(digitcaps)
    print("shape digitcaps", digitcaps.shape)
    masked = Mask()(digitcaps)
    print("shape masked:",masked.shape)

    out_flat = layers.Flatten()(masked)
    out_dense= layers.Dense(dense_unit, activation='relu',kernel_initializer=initializers.glorot_uniform(), input_dim= n_class*dim_capsule_caps2)(out_flat)
    out_drop= layers.Dropout(0.5)(out_dense)
    out_dense2= layers.Dense(dense_unit, activation='relu',kernel_initializer=initializers.glorot_uniform(), input_dim= n_class*dim_capsule_caps2)(out_drop)
    out = layers.Dense(2)(out_dense2)


    # Models for training and evaluation (prediction)
    train_model = models.Model(x, out) #masked_by_y
    eval_model = models.Model(x, out)
        
    return eval_model

def CPSmodel_radar(input_shape, n_class, routings, nb_filters, kernel_size, dim_capsule_caps1, num_capsule_caps1, dim_capsule_caps2, dense_unit):   

    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv2')(conv1)
    #from[20,4] to [18,256]
    print("Output 1st Conv: ", conv1.shape)
    
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=dim_capsule_caps1, n_channels=num_capsule_caps1, kernel_size=kernel_size, strides=2, padding='valid')
    
    print("primarycaps_out: ",primarycaps.shape)
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule_caps2, num_routing=routings,
                             name='digitcaps')(primarycaps)
   
    
    digitcaps = K.squeeze(digitcaps, axis=1)
    digitcaps = K.squeeze(digitcaps, axis=3)
    #digitcaps = layers.Reshape((n_class, dim_capsule_caps2))(digitcaps)
    print("shape digitcaps", digitcaps.shape)
    masked = Mask()(digitcaps)
    print("shape masked:",masked.shape)
    #masked = layers.Reshape((-1,n_class*dim_capsule_caps2))(masked)
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    # out_caps = Length(name='capsnet')(digitcaps) 
    out_flat = layers.Flatten()(masked)
    out_dense= layers.Dense(dense_unit, activation='relu',kernel_initializer=initializers.glorot_uniform(), input_dim= n_class*dim_capsule_caps2)(out_flat)
#     out_drop= layers.Dropout(0.5)(out_dense)
#     out_dense2= layers.Dense(dense_unit, activation='relu',kernel_initializer=initializers.glorot_uniform(), input_dim= n_class*dim_capsule_caps2)(out_flat)
    out = layers.Dense(2)(out_dense)


    # Models for training and evaluation (prediction)
    train_model = models.Model(x, out) #masked_by_y
    eval_model = models.Model(x, out)
        
    return eval_model

def CPSmodel_ir(input_shape, n_class, routings, nb_filters, kernel_size, dim_capsule_caps1, num_capsule_caps1, dim_capsule_caps2, dense_unit):   

    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv1')(x)
    #conv2 = layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv2')(conv1)
    #from[20,4] to [18,256]
    print("Output 1st Conv: ", conv1.shape)
    
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=dim_capsule_caps1, n_channels=num_capsule_caps1, kernel_size=kernel_size, strides=2, padding='valid')
    
    print("primarycaps_out: ",primarycaps.shape)
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule_caps2, num_routing=routings,
                             name='digitcaps')(primarycaps)
   
    
    digitcaps = K.squeeze(digitcaps, axis=1)
    digitcaps = K.squeeze(digitcaps, axis=3)
    #digitcaps = layers.Reshape((n_class, dim_capsule_caps2))(digitcaps)
    print("shape digitcaps", digitcaps.shape)
    masked = Mask()(digitcaps)
    print("shape masked:",masked.shape)
    #masked = layers.Reshape((-1,n_class*dim_capsule_caps2))(masked)
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    # out_caps = Length(name='capsnet')(digitcaps) 
    out_flat = layers.Flatten()(masked)
#     out_dense= layers.Dense(dense_unit, activation='relu',kernel_initializer=initializers.glorot_uniform(), input_dim= n_class*dim_capsule_caps2)(out_flat)
#     out_drop= layers.Dropout(0.5)(out_dense)
    out_dense2= layers.Dense(dense_unit, activation='relu',kernel_initializer=initializers.glorot_uniform(), input_dim= n_class*dim_capsule_caps2)(out_flat)
    out = layers.Dense(2)(out_dense2)


    # Models for training and evaluation (prediction)
    train_model = models.Model(x, out) #masked_by_y
    eval_model = models.Model(x, out)
        
    return eval_model
