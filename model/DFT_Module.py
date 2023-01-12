import tensorflow as tf
from tensorflow.keras.layers import *

def DFT_Module(a, DFT_number):
    a = Dense(DFT_number, activation='sigmoid')(a)
    
    a = BatchNormalization(axis=-1)(a) 
    
    a=Dropout(.3)(a)

    #FFT on time feature
    a = Permute((2, 1))(a)
    
    a = Lambda(lambda v: tf.signal.rfft(v))(a)

    #a = Permute((2, 1))(a)

    #real value
    a_real = Lambda(tf.math.real)(a)
    
    a_real = Activation('relu')(a_real)
    
    a_real = Dense(5, activation='sigmoid')(a_real)
    
    a_real = Flatten()(a_real)

    a_real = Dropout(.3)(a_real)
    
    return a_real