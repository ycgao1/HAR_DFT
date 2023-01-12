import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt


    
def matrix(predict, test_label):
    matrix=confusion_matrix(np.argmax(test_label, axis=1), np.argmax(predict, axis=1), normalize='true')
    matrix=np.round(matrix,4)
    
    plt.matshow(matrix, cmap=plt.cm.Blues) 
    #plt.colorbar()
    labels=['boxing','jack','jump','squats','walk']
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(len(matrix)): 
        for j in range(len(matrix)):
            plt.annotate(round(matrix[i,j],3), xy=(j, i), horizontalalignment='center', verticalalignment='center', color="white" if matrix[i, j] > 0.5 else "black")
    plt.ylabel('Human activity')
    plt.xlabel('Prediction label') 
    plt.savefig("results/confusion_matrix.png")
    
def draw_history(learning_hist):
    fig1, ax_acc = plt.subplots()
    plt.plot(learning_hist.history['accuracy'])
    plt.plot(learning_hist.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig("results/accuracy.png")
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss - cnn_lstm batch=3')
    plt.plot(learning_hist.history['loss'])
    plt.plot(learning_hist.history['val_loss'])
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig("results/loss.png")
    plt.show()
    np.savez("results/accurate", learning_hist.history['accuracy'],learning_hist.history['val_accuracy'])
    np.savez("results/loss", learning_hist.history['loss'],learning_hist.history['val_loss'])
    
def FLOPs(model):
    forward_pass = tf.function(
    model.call,
    input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    flops = graph_info.total_float_ops // 2
    print('Flops: {:,}'.format(flops))
    
