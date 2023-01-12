import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
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


        
def model_train(FeatureExt, DFT, DFT_number, frame, lr, beta_1, beta_2, checkpoint_monitor, checkpoint_mode,
                     batch_size, epochs, model_path, train_data, train_label, validation_data, validation_label): 
    
    model = classification(FeatureExt, DFT, DFT_number, frame)
    model.summary()
 
    adam = optimizers.Adam(lr, beta_1, beta_2, epsilon=1E-8,
                           decay=0.0, amsgrad=False)


    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                       optimizer=adam,
                      metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_path, monitor=checkpoint_monitor, verbose=1, 
                                 save_best_only=True, mode=checkpoint_mode)

    callbacks_list = [checkpoint]

    learning_hist = model.fit(train_data, train_label,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 shuffle=True,
                              validation_data=(validation_data,validation_label),
                              callbacks=callbacks_list
                              )
    del model
    return learning_hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--FeatureExt", choices=['FC', 'DFT_FC', 'Attention', 'CNN'], required = True)
    parser.add_argument("--DFT", default = 1, type=int)
    parser.add_argument("--DFT_number", default = None, type=int)
    parser.add_argument("--frame", default=60, type=int)
    parser.add_argument("--data_path", default='data/voxel/')
    parser.add_argument("--model_dir", default='model_data/')
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--checkpoint_monitor", default='val_accuracy', choices=['val_loss', 'val_accuracy'])
    parser.add_argument("--checkpoint_mode", default='max', choices=['max', 'min'])
    parser.add_argument("--batch_size", default=15, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--draw", default=1, type=int)
    
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
    print(model_name)
    model_path = model_dir+ model_name +'_model.h5'
    lr = args.learning_rate
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    batch_size = args.batch_size
    epochs = args.epochs
    checkpoint_monitor = args.checkpoint_monitor
    checkpoint_mode = args.checkpoint_mode
    draw = args.draw
    
    train_data, train_label = get_data(data_path, 'train')
    
    train_data, validation_data, train_label, validation_label = train_test_split(
    train_data, train_label, test_size=0.20, random_state=1)
    print('train_data:', train_data.shape)
    print('train_label', train_label.shape)
    print('validation_data:', validation_data.shape)
    print('validation_label', validation_label.shape)
    
    learning_hist = model_train(FeatureExt, DFT, DFT_number, frame, lr, beta_1, beta_2, checkpoint_monitor, checkpoint_mode,
                     batch_size, epochs, model_path, train_data, train_label, validation_data, validation_label)
    
    if(draw ==True):
        draw_history(learning_hist)
        
   
    
if __name__ == "__main__":
    main()
