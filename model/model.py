import sys
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from model.Feature_Extraction_Module import *
from model.DFT_Module import DFT_Module
from model.BiLSTM import BiLSTM

def classification(FeatureExt, DFT, DFT_number, frame):
    print('building the model ... ')
    inputs = Input(shape=(frame, 32, 32, 16,1))
    
    a = FC(inputs)
    if(FeatureExt == "FC"):
        a = FC(inputs)
    elif(FeatureExt == "DFT_FC"):
        a = DFT_FC(inputs)
    elif(FeatureExt == "Attention"):
        a = Attention_block(inputs)
    elif(FeatureExt == "CNN"):
        a = CNN(inputs)
    else:
        print("Model name wrong!")
        return
    
    #FFT on time feature
    if(DFT and (DFT_number != None)):
        a = DFT_Module(a, DFT_number)
    elif(not DFT):
        a = BiLSTM(a)
    else:
        print("DFT_number needed!")
        sys.exit()


    output = Dense(5, activation='softmax', name='output')(a)
    
    model = Model(inputs=[inputs], outputs=[output])

    return model