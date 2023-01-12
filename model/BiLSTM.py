from tensorflow.keras.models import *    
from tensorflow.keras.layers import *
    
def BiLSTM(a):
    a = BatchNormalization(axis=-1)(a)
    
    
    a=Dropout(.3)(a)
    
    a = Bidirectional(LSTM(16, return_sequences=False, stateful=False))(a)

    a = Dropout(.3)(a)
    
    return a