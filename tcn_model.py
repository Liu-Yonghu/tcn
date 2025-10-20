import keras
from keras import layers, models
from tcn_simple import TCN_model
from tcn_old import TCN
import tensorflow as tf
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

def model_TCN_simple(hidden,num_filters,k_size,dense, source, name='tcn'):
    if source == "ir":
        x = keras.Input(shape=(4, 64), name=f"{name}_input")
    elif source == "mmwave":
        x = keras.Input(shape=(12, 3), name=f"{name}_input")

    model = TCN_model(num_layers=hidden,
                      num_filters=num_filters,
                      kernel_size=k_size,
                      dilations=[2 ** i for i in range(hidden)],
                      output_len=1)
    tcn_out = model(x)
    print("shape_out_tcn:", tcn_out.shape)
    dense_out1 = layers.Dense(dense, name=f"{name}_dense_out_1")(tcn_out)
    dense_out = layers.Dense(2, name=f"{name}_dense_out_2")(dense_out1)
    print("dense_out:", dense_out.shape)
    model = models.Model(x, dense_out, name="tcn_model")
    return model

def model_TCN_complete(hidden, num_filters, k_size, dense, source, name='tcn'):
    if source == "ir":
        x = keras.Input(shape=(4, 64), name=f"{name}_input")
    elif source == "mmwave":
        x = keras.Input(shape=(12, 3), name=f"{name}_input")
    tcn_out = TCN(nb_filters=num_filters,
                  kernel_size=k_size,
                  nb_stacks=1,
                  dilations=[2 ** i for i in range(hidden)],
                  padding='same',
                  use_skip_connections='True',
                  dropout_rate=0.01,
                  return_sequences=False,
                  activation='relu',
                  kernel_initializer='glorot_uniform',
                  use_layer_norm=True,
                  name='tcn')(x)
    flatten_out = layers.Flatten()(tcn_out)
    dense_out1 = layers.Dense(dense,name= f"{name}_dense_out_1")(flatten_out)
    dense_out = layers.Dense(2, name= f"{name}_dense_out_2")(dense_out1)
    print("dense_out:", dense_out.shape)
    model = models.Model(x, dense_out,name="tcn_model")
    return model