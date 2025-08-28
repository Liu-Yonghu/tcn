import inspect
from typing import List

import tensorflow as tf
# pylint: disable=E0611,E0401
from tensorflow.keras import backend as K, Model, Input, optimizers
# pylint: disable=E0611,E0401
from tensorflow.keras import layers
# pylint: disable=E0611,E0401
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
# pylint: disable=E0611,E0401
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization
# Funzione per ruotare il kernel lungo i canali (ruotando di 90° per volta)

# Modello TCN con più layer di convoluzione equivariante
class TCN_model(tf.keras.Model):
    def __init__(self, num_layers, num_filters, kernel_size, dilations, output_len=1, dropout_rate=0.05, name='tcn'):
        super(TCN_model, self).__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilations = dilations

        self.output_len = output_len
        
        self.conv_layers = []
        self.norm_layers = []
        self.act_layers = []
        self.dropout_layers = []
        
        # Creiamo più layer di convoluzione equivariante
        for i in range(num_layers):
            self.conv_layers.append(layers.Conv1D(
                filters=num_filters, 
                kernel_size=kernel_size, 
                dilation_rate=dilations[i],
                padding='same',
                name=f"{name}_conv_layer_{i}"
            ))
            # Aggiungiamo un LayerNormalization dopo ogni layer di convoluzione
            self.norm_layers.append(layers.LayerNormalization(axis=-1, name=f"{name}_norm_layer_{i}"))
            self.act_layers.append(layers.Activation("relu", name= f"{name}_activation_layer_{i}"))
            # Aggiungiamo SpatialDropout1D
            if dropout_rate > 0:
                self.dropout_layers.append(layers.SpatialDropout1D(rate=dropout_rate, name=f"{name}_dropout_layer_{i}"))
            else:
                self.dropout_layers.append(None)  # Placeholder per mantenere la lista


        
        
        # Strato finale convoluzionale, mantenendo tutti i filtri e le rotazioni
        self.conv_out = layers.Conv1D(filters=self.num_filters, kernel_size=1,name=f"{name}_conv_out")  # Output layer finale
        # Aggiungiamo un LayerNormalization prima dell'output finale, opzionale
        self.final_norm = layers.LayerNormalization(axis=-1,name=f"{name}_final_norm")


    def call(self, inputs):
        # Aggiungiamo una dimensione per i canali (da (batch_size, 15, 4) a (batch_size, 15, 4, 1))
        # x = tf.expand_dims(inputs, -1)
        x = inputs
        
        # Applichiamo i layer di convoluzione equivariante
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)  # Convoluzione
            x = self.norm_layers[i](x)    # Normalizzazione
            x = self.act_layers[i](x)      # Attivazione
            if self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x)  # Dropout se presente


        
        # Strato di output
        x = self.conv_out(x)
        # Applicare la normalizzazione finale, se desiderato
        x = self.final_norm(x)
        #print(x.shape)
        output_slice_index = K.shape(x)[1] // 2
        #print(output_slice_index)
        x = x[:, output_slice_index, :]  # Estrai il valore intermedio della sequenza
        #print(x.shape)
        return x
    
    def get_config(self):
        # Serializzazione del modello, non chiamare super() in tf.keras.Model
        return {
            'num_layers': self.num_layers,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'dilations': self.dilations,
            'output_len': self.output_len
        }

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]  # Il batch size rimane invariato
        time_steps = input_shape[1]  # Dimensione temporale di input (rimane invariata)
        # L'output avrà num_filters * num_rotations feature maps
        return (batch_size, 1, self.num_filters)