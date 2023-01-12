import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
sub_dirs=['boxing','jack','jump','squats','walk']

import glob
import os
import numpy as np
# random seed.
rand_seed = 200
from numpy.random import seed
seed(rand_seed)
#from tensorflow import set_random_seed
#set_random_seed(rand_seed)
from tensorflow.random import set_seed
set_seed(rand_seed)
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
#from tensorflow.keras.layers.convolutional import *
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import *
from tensorflow.keras.utils import plot_model
from model.model import classification
import matplotlib.pyplot as plt
from data_loader import get_data

from metrics import *
import argparse

def model_test(FeatureExt, DFT, DFT_number, frame, model_name, model_path, test_data, test_label, confusion, FLOPS_calculation):
    print("test:")
    model = classification(FeatureExt, DFT, DFT_number, frame)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights(model_path, by_name=True)

    result=model.evaluate(test_data, test_label, verbose=2)
    print("Accuracy:", result[1])
    predict=model.predict(test_data, batch_size=200)
    print(predict.shape)
    if(confusion):
        matrix(predict, test_label)
    if(FLOPS_calculation):
        FLOPS(model)
    
    del model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--FeatureExt", choices=['FC', 'DFT_FC', 'Attention', 'CNN'], required = True)
    parser.add_argument("--DFT", default = 1, type=int)
    parser.add_argument("--DFT_number", default = None, type=int)
    parser.add_argument("--frame", default=60, type=int)
    parser.add_argument("--data_path", default='data/voxel/')
    parser.add_argument("--model_dir", default='model_data/')
    parser.add_argument("--confusion_metrics", default=1, type=int)
    parser.add_argument("--FLOPS", default=1, type=int)
    
    args = parser.parse_args()
    
    frame = args.frame
    data_path = args.data_path
    model_dir = args.model_dir
    FeatureExt = args.FeatureExt
    DFT = args.DFT
    DFT_number = args.DFT_number
    if(DFT and DFT_number != None):   
        model_name = FeatureExt+"_DFT"
    elif(DFT and DFT_number == None):
        print("DFT number is needed!")
        return
    elif(DFT_number == None):
        model_name = FeatureExt+"_BiLSTM"
    else:
        print("DFT number is not needed!")
        return
    model_path = model_dir+ model_name +'_model.h5'
    confusion = args.confusion_metrics
    FLOPS_calculation= args.FLOPS
    
    
    test_data, test_label = get_data(data_path, 'test')
     
    model_test(FeatureExt, DFT, DFT_number, frame, model_name, model_path, test_data, test_label, confusion, FLOPS_calculation)
    
if __name__ == "__main__":
    main()
