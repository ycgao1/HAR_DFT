sub_dirs=['boxing','jack','jump','squats','walk']

import numpy as np
import tensorflow as tf

def one_hot_encoding(y_data, sub_dirs, categories=5):
    Mapping = dict()

    count = 0
    for i in sub_dirs:
        Mapping[i] = count
        count = count+1

    y_features2 = []
    for i in range(len(y_data)):
        Type = y_data[i]
        lab = Mapping[Type]
        y_features2.append(lab)

    y_features = np.array(y_features2)
    y_features = y_features.reshape(y_features.shape[0], 1)
    from tensorflow.keras.utils import to_categorical
    y_features = to_categorical(y_features, categories)
    del y_data
    
    return y_features

def get_data(extract_path, load_type):
    data=[]
    label=[]
    i=0
    if(load_type == "train"):
        extract_path=extract_path+'Train/'
    elif(load_type == "test"):
        extract_path=extract_path+'Test/'
    else:
        print("load type wrong!")
        return
    for sub_dir in sub_dirs:
        Data_path = extract_path+sub_dir
        data_raw=np.load(Data_path+'.npz')
        data_n = np.array(data_raw['arr_0'],dtype=np.dtype(np.float32))
        data_n = data_n.reshape(data_n.shape[0],data_n.shape[1], data_n.shape[2],data_n.shape[3],data_n.shape[4],1)
        label_n = one_hot_encoding(data_raw['arr_1'], sub_dirs, categories=5)
        if i==0:
            data=data_n
            label=label_n
        else:
            data = np.concatenate((data, data_n), axis=0)
            label = np.concatenate((label, label_n), axis=0)
        i+=1

        del data_raw, 
        del data_n, label_n
    return data, label