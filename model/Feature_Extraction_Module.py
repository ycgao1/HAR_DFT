import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

def attention_3d_block(inputs):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if False:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def FC(inputs):
    a = TimeDistributed(Flatten())(inputs)
    
    return a
    
def DFT_FC(inputs):
    a = TimeDistributed(Lambda(lambda v: tf.signal.rfft(v)))(inputs)
                        
    a_real = Lambda(tf.math.real)(a)
    
    a = LayerNormalization(axis=[1, 2, 3])(a_real)
    
    a=TimeDistributed(Flatten())(a)
    
    return a

def Attention_block(inputs):
    a=TimeDistributed(Flatten())(inputs)
    
    a = Dense(128, activation='sigmoid')(a)
    
    a = attention_3d_block(a)
    a = BatchNormalization(axis=-1)(a)

    
    return a

def CNN(inputs):
    # 1st layer group
    layer1=TimeDistributed(Conv3D(4, (3, 3, 3), strides=(1, 1, 1), name="conv1a", padding="same", activation="relu"))(inputs)
    # 2nd layer group
    layer2=TimeDistributed(Conv3D(8, (3, 3, 3), strides=(1, 1, 1), name="conv1b", padding="same", activation="relu"))(layer1)

    layer2=TimeDistributed(MaxPooling3D(name="pool1", strides=(2, 2, 2), pool_size=(2, 2, 2), padding="valid"))(layer2)
    layer2 = BatchNormalization(axis=5)(layer2)

    # 3rd layer group
    layer3=TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2a", padding="same", activation="relu"))(layer2)
    layer3=TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2b", padding="same", activation="relu"))(layer3)
    layer3=TimeDistributed(MaxPooling3D(strides=(2, 2, 2), pool_size=(2, 2, 2), name="pool2", padding="valid"))(layer3)

    layer3=TimeDistributed(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), name="conv3a", padding="same", activation="relu"))(layer3)
    layer3=TimeDistributed(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), name="conv3b", padding="same", activation="relu"))(layer3)
    layer3=TimeDistributed(MaxPooling3D(strides=(2, 2, 2), pool_size=(2, 2, 2), name="pool3", padding="valid"))(layer3)
    layer3 = BatchNormalization(axis=-1)(layer3)

    a=TimeDistributed(Flatten())(layer3)
    return a