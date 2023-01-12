import glob
import os
import numpy as np
import sys
import argparse

sub_dirs=['boxing','jack','jump','squats','walk']

def coordinate2array(x, y, z, intensity, x_point, y_point, z_point, intensity_level,  x_min, x_max, y_min, y_max, z_min, z_max, intensity_max, intensity_min):

    # change
    
    X = np.round((x_point-1)*(x-x_min)/(x_max-x_min)).astype(int)
    Y = np.round((y_point-1)*(y-y_min)/(y_max-y_min)).astype(int)
    Z = np.round((z_point-1)*(z-z_min)/(z_max-z_min)).astype(int)

    pixel = np.zeros([x_point, y_point, z_point])
    pixel_i = np.zeros([x_point, y_point, z_point])

    for i in range(len(X)):
        x_c = X[i]
        y_c = Y[i]
        z_c = Z[i]
        pixel[x_c, y_c, z_c] = pixel[x_c, y_c, z_c]+1
        pixel_i[x_c, y_c, z_c] = pixel_i[x_c, y_c, z_c]+intensity[i]
        pixel_I = np.zeros([x_point, y_point, z_point])
    idNonZeros = np.where(pixel != 0)
    idZeros = np.where(pixel == 0)
    pixel_I[idZeros] = 0
    pixel_I[idNonZeros] = pixel_i[idNonZeros]/pixel[idNonZeros]
    pixel_I = np.round((intensity_level-1)*(pixel_I-intensity_min)/(intensity_max-intensity_min)).astype(int)

    del x, y, z, intensity, pixel, pixel_i

    return pixel_I


def get_data(file_path, total_frame, sliding):

    with open(file_path) as f:
        lines = f.readlines()

    wordlist = []
    for line in lines:
        for word in line.split():
            wordlist.append(word)

    # extract x,y,z
    frame_num_count = -1
    frame_num = []
    x = []
    y = []
    z = []
    intensity = []
    for i in range(len(wordlist)):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_num_count += 1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])
        if wordlist[i] == "intensity:":
            intensity.append(wordlist[i+1])

    # list2array
    frame_num = np.asarray(frame_num)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    intensity = np.asarray(intensity)

    # datatype
    frame_num = frame_num.astype(int)
    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    intensity = intensity.astype(float)
    
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    z_max = np.max(z)
    z_min = np.min(z)
    intensity_max = np.max(intensity)
    intensity_min = np.min(intensity)

    # print(len(frame_num))
    # frame
    inten_test=[]
    x_test=[]
    y_test=[]
    z_test=[]
    data = dict()
    for i in range(len(frame_num)):
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i], y[i], z[i], intensity[i]])
        else:
            data[frame_num[i]] = []
            data[frame_num[i]].append([x[i], y[i], z[i], intensity[i]])
        if(frame_num[i]==212):
            x_test.append(x[i])
            y_test.append(y[i])
            z_test.append(z[i])
            inten_test.append(intensity[i])

    # dict2pixels
    pixels = []
    for i in data:
        data2 = data[i]
        data2 = np.asarray(data2)

        x_c = data2[:, 0]
        y_c = data2[:, 1]
        z_c = data2[:, 2]
        intensity_c = data2[:, 3]
 
        pix = coordinate2array(x_c, y_c, z_c, intensity_c, 32, 32, 16, 512, x_min, x_max, y_min, y_max, z_min, z_max, intensity_max, 0)
        pixels.append(pix)
    pixels = np.asarray(pixels)

    # sild window
    # total_frame=60;
    # silding=15;
    i = 0
    data_sildwindow = []
    while (i+total_frame) <= pixels.shape[0]:
        data_sildwindow.append(pixels[i:i+total_frame, :, :, :])
        i = i+sliding

    if pixels.shape[0] % sliding != 0:
        data_sildwindow.append(
            pixels[pixels.shape[0]-total_frame:pixels.shape[0], :, :, :])

    data_sildwindow = np.asarray(data_sildwindow)
    # return data, pixels

    del frame_num, x, y, z, intensity, wordlist, pixels, data,

    return data_sildwindow


def parse_RF_files(parent_dir, sub_dirs, total_frame, sliding, file_ext='*.txt'):
    print(sub_dirs)
    features = np.empty((0, total_frame, 32, 32, 16))
    labels = []

    for sub_dir in sub_dirs:
        files = sorted(glob.glob(os.path.join(parent_dir, sub_dir, file_ext)))
        for fn in files:
            print(fn)
            print(sub_dir)
            train_data = get_data(fn, total_frame, sliding)
            features = np.vstack([features, train_data])

            for i in range(train_data.shape[0]):
                # labels.append(action[sub_dir])
                labels.append(sub_dir)
            print(features.shape, len(labels))

            del train_data
            # gc.collect()
    labels = np.asarray(labels)

    return features, labels


def extract(frame, sliding):
    print(type)
    parent_dir = 'data/Train/'
    sub_dirs = ['boxing', 'jack', 'jump', 'squats', 'walk']
    extract_path = 'extract/Train/'

    for sub_dir in sub_dirs:
        feature, label = parse_RF_files(
            parent_dir, [sub_dir], frame, sliding)
        
        Data_path = extract_path + sub_dir
        if feature.shape[0] == 0:
            print("no ", sub_dir, " files")
            continue
        np.savez(Data_path, feature, label)
        
        print(sub_dir, feature.shape)
        del feature, label
        
    parent_dir = 'data/Test/'
    sub_dirs = ['boxing', 'jack', 'jump', 'squats', 'walk']
    extract_path = 'extract/Test/'

    for sub_dir in sub_dirs:
        feature, label = parse_RF_files(
            parent_dir, [sub_dir], frame, sliding)
        
        Data_path = extract_path + sub_dir
        if feature.shape[0] == 0:
            print("no ", sub_dir, " files")
            continue
        np.savez(Data_path, feature, label)
        
        print(sub_dir, feature.shape)
        del feature, label
def extract(frame, sliding, parent_dir, voxel_path):
    
    for sub_dir in sub_dirs:
        features, labels = parse_RF_files(parent_dir+'Train/',[sub_dir], frame, sliding)
        Data_path = voxel_path + 'Train/' + sub_dir
        if features.shape[0]==0:
            print("no ",sub_dir," files")      
        else:
            np.savez(Data_path, features,labels)
            print(sub_dir, features.shape)
        del features, labels
        
    for sub_dir in sub_dirs:
        features, labels = parse_RF_files(parent_dir+'Test/',[sub_dir], frame, sliding)
        Data_path = voxel_path + 'Test/' + sub_dir
        if features.shape[0]==0:
            print("no ",sub_dir," files")      
        else:
            np.savez(Data_path, features,labels)
            print(sub_dir, features.shape)
        del features, labels

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=60, type=int)
    parser.add_argument("--sliding_window", default=10, type=int)
    parser.add_argument("--data_path", default='data/raw/')
    parser.add_argument("--data_save", default='data/voxel/')
    args = parser.parse_args()
    
    parent_dir = args.data_path
    voxel_path = args.data_save
    frame = args.frame
    sliding = args.sliding_window

    return extract(frame, sliding, parent_dir, voxel_path)
    
    
if __name__ == "__main__":
    main()