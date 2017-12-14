'''
This is the Keras implementation of
'Domain-Adversarial Training of Neural Networks' by Y. Ganin
This allows domain adaptation (when you want to train on a dataset
with different statistics than a target dataset) in an unsupervised manner
by using the adversarial paradigm to punish features that help discriminate
between the datasets during backpropagation.
This is achieved by usage of the 'gradient reversal' layer to form
a domain invariant embedding for classification by an MLP.
The example here uses the 'MNIST-M' dataset as described in the paper.
Credits:
- Clayton Mellina (https://github.com/pumpikano/tf-dann) for providing
  a sketch of implementation (in TF) and utility functions.
- Yusuke Iwasawa (https://github.com/fchollet/keras/issues/3119#issuecomment-230289301)
  for Theano implementation (op) for gradient reversal.
Author: Vanush Vaswani (vanush@gmail.com)
'''

from __future__ import print_function
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, BatchNormalization, Activation
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adam
from keras.models import Model
#from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.datasets import mnist
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.manifold import TSNE

#from keras.layers import GradientReversal
from Gradient_Reverse_Layer import GradientReversal
#from keras.engine.training import make_batches
#from keras.datasets import mnist_m
from keras import backend
backend.set_image_data_format('channels_first')
from keras.models import load_model
   
           
def train_loader(start, end, filename):
    #print(start, end)
    x_batch = HDF5Matrix(filename,
                             'my_data', start=start, end=end)
    x_batch = np.stack([x_batch], axis=2)
    #x_batch = np.array(x_batch)
    y_batch = HDF5Matrix(filename,
                         'my_labels', start=start, end=end)
    return x_batch, y_batch
 
 
# ------------------------------------------------------------------------------
# Funktion for Callbacks in train_on_batch
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        histogram = tf.summary.histogram
        # tf.summary.FileWriter('logs/', histogram)
        callback.writer.add_summary(summary, batch_no)
        # callback.writer.add_summary(histogram, batch_no)
        callback.writer.flush()
# ------------------------------------------------------------------------------
# Stuff for Tensorboard callbacks

"""https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py"""
        
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    # ------------------------------------------------------------------------------                     

class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = Adam()

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Convolution1D(nb_filters, nb_conv,
                            border_mode='valid', name = 'Conv1')(model_input)
        net = BatchNormalization(axis = 1)(net)
        net = Activation('relu')(net)
#        net = Dropout(0.5)(net)
#        net = Convolution1D(nb_filters, nb_conv,
#                            border_mode='valid',
#                            activation='relu', name = 'Conv1')(model_input)
#        net = MaxPooling1D(pool_size=nb_pool)(net)        
#        net = Convolution1D(nb_filters, nb_conv,
#                            border_mode='valid',
#                            activation='relu', name = 'Conv2')(net)
#        # net = Convolution1D(nb_filters, nb_conv,
#                            # activation='relu')(net)
#        net = BatchNormalization()(net)
        net = MaxPooling1D(pool_size=nb_pool)(net)
        net = Convolution1D(nb_filters, nb_conv,
                            border_mode='valid', name = 'Conv1')(model_input)
        net = BatchNormalization(axis = 1)(net)
        net = Activation('relu')(net)
        net = MaxPooling1D(pool_size=nb_pool)(net)
        net = Flatten()(net)
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(128, activation='relu', name = 'Dense1')(model_input)
#        net = BatchNormalization()(net)
        net = Dropout(0.5)(net)
        net = Dense(128, activation='relu', name = 'Dense2')(net)
        net = Dropout(0.5)(net)
        net = Dense(nb_classes, activation='softmax',
                    name='classifier_output')(net)
        return net

    def build_source_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=net)
        if plot_model:
            plot(model, show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_dann_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input) # this is model until Flatten()
        self.grl = GradientReversal(1.0) # add GradientReversal
        branch = self.grl(net) 
        branch = Dense(128, activation='relu')(branch)
#        branch = BatchNormalization()(branch)
        branch = Dropout(0.5)(branch)
        branch = Dense(128, activation='relu')(branch)
#        branch = BatchNormalization()(branch)
        branch = Dropout(0.5)(branch)
        branch = Dense(2, activation='softmax', name='domain_output')(branch) # add feed forward part

        # When building DANN model, route first half of batch (source examples)
        # to domain classifier, and route full batch (half source, half target)
        # to the domain classifier.
        net = Lambda(lambda x: K.switch(K.learning_phase(),
                     x[:int(batch_size // 2), :], x),
                     output_shape=lambda x: ((batch_size // 2,) +
                     x[1:]) if _TRAIN else x[0:])(net)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=[branch, net])
        if plot_model:
            plot(model, show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy',
                      'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics={'classifier_output': 'accuracy', 'domain_output' : 'accuracy'})
        return model

    def build_tsne_model(self, main_input):
        '''Create model to output intermediate layer
        activations to visualize domain invariant features'''
        tsne_model = Model(input=main_input,
                           output=self.domain_invariant_features)
        return tsne_model

def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
def batch_generator(len_x_train, id_array, batch_size, data, labels):
    np.random.shuffle(id_array)  # shuffling is fullfilled here
    # not sure about repeated shuffling, because we should hit every sample once
    # probably better to put this function before loop.
    for start in range(0, len_x_train, batch_size):
        end = min(start + batch_size, len_x_train)
        batch_ids = id_array[start:end]
        if labels is not None:
            x_batch = data[batch_ids]
            y_batch = labels[batch_ids]  # this only works if labels is numpy array
            yield x_batch, y_batch
        else:
            x_batch = data[batch_ids]
            yield x_batch
# ----------------------------------------------------------------
# Set Parameters
batch_size = 500
batch_size_test = 500
nb_epoch = 5000
nb_classes = 2                  # number of classes
input_dim = 924                  # input size
nb_filters = 32                 # number of filters in Conv1D
nb_pool = 2                     # pool size of Conv1D
nb_conv = 3                     # Size of receptive field
_TRAIN = K.variable(1, dtype='uint8')
loading = True
train_source_only = False
train_dann = False 
evaluate_model = True
extract_latent_variable = False
analyze_predictions = False
unsupervised = False
make_movie = False
L = 12

# ----------------------------------------------------------------
# Prepare Data
# We dont have to shuffle, we do it in train_generator
len_x_train = 10000
len_x_target = 10000
rescaler = 0

folder = 'Data/'
img_folder = 'IMAGES_TRAINING/'

train_weight_path = folder + 'weights_MBL_'
test_weight_path = folder + 'weights1100.npy'
unsuperivised_weight_path = folder + 'unsupervised_MBL_'+str(len_x_train)+'.npy'
pred_target_path = folder+'predictions_on_target_MBL_'+str(len_x_train)
pred_source_path = folder+'predictions_on_train_MBL_'+str(len_x_train)


if loading:
    path_source = folder+'00SOURCE' + '_N' + str(L) + '.h5'
    path_target = folder+'00TARGET' + '_N' + str(L) + '.h5'
    #path_target = 'TARGET_ABSOLUTE_OBC_disorder_w1_1.0_w2_05w1_not_shuffled_full_N64.h5'
    x_train, y_train = train_loader(0+3000, len_x_train+3000, path_source)
    x_train_verify = x_train
    y_train = np.array(y_train)
    print('Source SET loaded')
    x_target, y_target = train_loader(0, len_x_target, path_target)
    x_target_verify = x_target
    del y_target
    print('TARGET SET loaded')
    len_x_target = len(x_target)
    len_x_train = len(x_train)
    source_index_arr = np.arange(x_train.shape[0])
    target_index_arr = np.arange(x_target.shape[0])
    # This is to reduce the data set for first tests
    # np.random.shuffle(source_index_arr)
    # source_index_arr = source_index_arr[0:5000]
#    np.random.shuffle(target_index_arr)
#    target_index_arr = target_index_arr[0:5000]
#    x_train, y_train = x_train[source_index_arr], y_train[source_index_arr]
#    x_target = x_target[target_index_arr]
#    len_x_train = 5000
#    len_x_target = 5000
#    source_index_arr = np.arange(x_train.shape[0])
#    target_index_arr = np.arange(x_target.shape[0])
# ------------------------------------------------------------------
# Build models
main_input = Input(shape=(input_dim, 1), name='main_input')

builder = DANNBuilder()

# These are separate models
src_model = builder.build_source_model(main_input) # this is CNN and feature extractor
src_vis = builder.build_tsne_model(main_input) # this is CNN until Flatten

dann_model = builder.build_dann_model(main_input)
dann_vis = builder.build_tsne_model(main_input)

# --------------------------------------------------------------------
# Train SOURCE ONLY

if train_source_only:
    print('Training source only model')

    src_model.fit(x_train, y_train, batch_size=64, nb_epoch=50, verbose=1)
    # shuffle = 'batch' is used for HDF5 files
    print('Evaluating target samples on source-only model')
    print('Accuracy: ', src_model.evaluate(x_target, y_target)[1])
# -------------------------------------------------------------------


batches_per_epoch = len_x_train / batch_size
num_steps = nb_epoch * batches_per_epoch
j = 0
loss_label = []
loss_domain = []
metrics_list = []   

# ------------------------------------------------------------------------------
# Stuff for Callbacks
log_path = './logs'
train_names = ['train_loss', 'train_mae'] # Callback names for Train Set
val_names = ['val_loss', 'val_mae'] # Callback names for Validation Set
# ------------------------------------------------------------------------------
        
if train_dann:   
    print('Training DANN model')
    jj = 0 
    for i in range(nb_epoch):
        
        # batches = make_batches(X_train.shape[0], batch_size / 2)
        # target_batches = make_batches(XT_train.shape[0], batch_size / 2)

        src_gen = batch_generator(len_x_train, source_index_arr, batch_size // 2, x_train, y_train)
        target_gen = batch_generator(len_x_target, target_index_arr, batch_size // 2, x_target, None)
            
        losses = list()
        acc = list()

        print('Epoch ', i)
#        if i == 0:
#            weight = np.load(folder + 'weights44.npy')
#            dann_model.set_weights(weight)
#        else:
#            np.save(folder+'weights'+str(i), dann_model.get_weights()) # save weights after each epoch
        if i%10 == 0:
            np.save(folder+'weights'+str(i), dann_model.get_weights()) # save weights after each epoch
        device = "gpu"
        with tf.device("/" + device + ":0"):
            for (xb, yb) in src_gen:
                # Update learning rate and gradient multiplier as described in
                # the paper.
                p = float(j) / num_steps
                l = (2. / (1. + np.exp(-10. * p)) - 1)
                lr = 0.0005 / (1. + 10 * p)**0.75 #0.00005 / (1. + 10 * p)**0.75 # # 0.01 / (1. + 10 * p)**0.75 # 
                builder.grl.hp_lambda = 5.0 #0.00001
                builder.opt.lr = lr
                if xb.shape[0] != batch_size // 2:
                    #print('1')
                    continue
                try:
                    xt = next(target_gen)#target_gen.next()
                    #print(xt[0])
                    #print('2')
                except:
                    # Regeneration
                    target_gen = target_gen(len_x_target, target_index_arr, batch_size // 2, x_target, None)
                    #print('3')
                #print(xb.shape, xt.shape)
                # Concatenate source and target batch
                domain_labels = np.vstack([np.tile([0, 1], [len(xb), 1]),
                                   np.tile([1., 0.], [len(xt), 1])])
                # domain_labels = to_categorical(domain_labels)
                xb = np.vstack([xb, xt])

                print(j, jj)
                #print('xb', len(xb), 'yb', len(yb), 'domain', len(domain_labels))
#                metrics = dann_model.train_on_batch({'main_input': xb},
#                                                    {'classifier_output': yb,
#                                                    'domain_output': domain_labels}, check_batch_dim=False)
                metrics = dann_model.train_on_batch({'main_input': xb},
                                                    {'classifier_output': yb,
                                                    'domain_output': domain_labels}, check_batch_dim=False)
                metrics_list.append([metrics[0], metrics[1], metrics[2]])
                print(metrics[1], metrics[2])
                np.save('metrics', metrics_list)
                
                #callback = TensorBoard(log_dir=log_path, histogram_freq=1, write_images = True, embeddings_freq = 1, embeddings_layer_names=['Conv1'],  write_graph=False)
                #callback.set_model(dann_model)
                #write_log(callback, train_names, metrics, j)
                
                # METRICS names['loss','domain_output_loss','classifier_output_loss','domain_output_acc','classifier_output_acc']
#                if j%10 == 0:
#                    yb = np.vstack([yb, np.tile([0., 0.], [len(yb), 1]) ])
#                    a = dann_model.test_on_batch({'main_input': xb},{'classifier_output': yb,'domain_output': domain_labels}, sample_weight=None)
#                    out = dann_model.predict_on_batch(xb)
#                    y_pred = out[1]
#                    domain_pred = out[0]
#                    # out[0] is domain output, out[1] is classifier out
#                    mean_square_label = 1/len(yb)*np.sum( (y_pred[i]-yb[i])**2 for i in range(len(yb)))
#                    mean_square_domain = 1/len(yb)*np.sum( (domain_pred[i]-domain_labels[i])**2 for i in range(len(yb)))
#                    yb = tf.convert_to_tensor(yb, dtype=tf.float64)
#                    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float64)
#                    K.categorical_crossentropy(yb, y_pred)
#                    print(mean_square_label)
#                    print(mean_square_domain)
#                    loss_domain.append(mean_square_domain)
#                    loss_label.append(mean_square_label)
                j += 1
            jj += 1
            # metrics_list.append([metrics[0], metrics[1], metrics[2]])
            if i%1 == 0:
                test = dann_model.predict(x_train_verify)
                test2 = dann_model.predict(x_target_verify)
                plt.clf()
                plt.plot(test[1])
                plt.savefig(img_folder+'01train_epoch'+str(jj))
                plt.clf()
                plt.plot(test[0])
                plt.savefig(img_folder+'03DOMAIN_TRAIN_epoch'+str(jj))
                plt.clf()
                plt.plot(test2[1])
                plt.savefig(img_folder+'02target_epoch'+str(jj))
                plt.clf()
                plt.plot(test2[0])
                plt.savefig(img_folder+'04DOMAIN_target_epoch'+str(jj))
                

if evaluate_model:
    
    # for evaluation we have to increase batch size
    
    weight = np.load(test_weight_path) # reload weights
    dann_model.set_weights(weight)
    dann_model.summary()
    # np.random.shuffle(source_index_arr)
    # np.random.shuffle(target_index_arr)
    # x_train = x_train[source_index_arr]
    # y_train = y_train[source_index_arr]
    # x_target = x_target[target_index_arr]
    # x_axis = np.linspace(0,1,len(x_target))[target_index_arr]
    print('Evaluating target samples on DANN model')
    #out1 = dann_model.predict_on_batch(x_train[0:batch_size // 2])
    #target_index_arr = np.arange(x_target.shape[0])
    #index = np.random.shuffle(target_index_arr)
    list_target = []
    list_train = []
    hmax_list = np.linspace(0.9, 5.0, 20)
    epsilon_list = epsilon_list = np.linspace(0.1, 0.9, 9)
    matrix = np.zeros((len(epsilon_list),len(hmax_list)))
    for k in range(len(epsilon_list)):
        epsilon = epsilon_list[k]
        for l in range(len(hmax_list)): #100_samp_EVALUATE_N12eps0.9hmax0.9
            print(k,l)
            val = 0.0
            hmax = hmax_list[l]
            filename = 'EVALUATION_FILES/' + '00EVALUATE_N' + str(L) + 'eps'+ str(epsilon) + 'hmax' + str(hmax) +'.h5'
            x_batch = HDF5Matrix(filename, 'my_data')
            x_batch = np.stack([x_batch], axis=2)
            test2 = dann_model.predict(x_batch)
            for i in test2[1]:
                val += i[0]
            matrix[k][l] = val/len(test2[1])
    np.save('10_Dez_raw_matrix_weight_1150', matrix)
            
    # matrix = np.load('2D_plot_raw_matrix.npy')
    import copy
    matrix2 = copy.deepcopy(matrix)
#    matrix2 = matrix[:]                          
    for i in range(9):
         for j in range(20):
             if matrix2[i][j]>0.2:
                 matrix2[i][j] = 1.0
             else:
                 matrix2[i][j] = 0.0
      
    plt.clf()              
    plt.pcolormesh(matrix2)  
    plt.savefig('test01.pdf')
    matrix = np.load('10_Dez_raw_matrix_weight_1150.npy')
    plt.clf()              
    plt.pcolormesh(matrix)  
    plt.savefig('test02.pdf')






if analyze_predictions:
    list_target = np.load( pred_target_path+'.npy') #40001 states
    list_train = np.load( pred_source_path+'.npy')  #40001 states
    train_phase = []
    target_phase = []
    for j in range(0, len_x_train // batch_size_test):
        for i in range(batch_size_test):
            train_phase.append(list_train[j][1][i][0])
            target_phase.append(list_target[j][1][i][0])
    average = 5000 # this depends on how states were produced. We have 10 disorder realizations and 50 states around epsilon
    hmin, hmax = 0.9, 4.7
    avg_list = []
    for i in range(20): #because we have 20 disorder realizations
        avg_list.append(np.sum(target_phase[i*average:(i+1)*average])/average)     
    plt.clf()
    plt.plot(np.linspace(0,1,len(train_phase)), train_phase)
    plt.savefig(folder+'train')
    #a, b = zip(*sorted(zip(x_axis[:len(target_phase)], target_phase)))
    # sort shuffled data for plot
    plt.clf()
    plt.plot(np.linspace(0,1,len(train_phase)), target_phase)
    plt.savefig(folder+'target')
    plt.clf()
    plt.plot(np.linspace(hmin, hmax, len(avg_list)), avg_list)
    plt.savefig(folder+'AVERAGED_target')
   

if extract_latent_variable:
    weight = np.load(test_weight_path) # reload weights
    dann_model.set_weights(weight)
    combined_test_imgs = x_target
    dann_embedding = dann_vis.predict(combined_test_imgs)
    np.save('LATENT_SPACE_OUTPUT_DANN', dann_embedding)


if unsupervised:
    # Created mixed dataset for TSNE visualization
    num_test = 1000
    np.random.shuffle(source_index_arr)
    np.random.shuffle(target_index_arr)
    x_train = x_train[source_index_arr]
    y_train = y_train[source_index_arr]
    x_target = x_target[source_index_arr]#x_target[target_index_arr]
    combined_test_imgs = np.vstack([x_train[:num_test], x_target[:num_test]])
    combined_test_labels = np.vstack([y_train[:num_test], y_train[:num_test]])
    combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                     np.tile([0., 1.], [num_test, 1])])
    weight = np.load(test_weight_path) # reload weights
    dann_model.set_weights(weight)
    #dann_vis.predict(x_train[0:200])
    """
    dann_embedding = dann_vis.predict(combined_test_imgs)
    dann_tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=3000)
    tsne = dann_tsne.fit_transform(dann_embedding)
    #argmax(1) makes [0,1] to 1 and [1,0] to 0
    plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'DANN')
    plt.savefig('DANN_tSNE')"""
    
    # ----------------------------
    # Test look only at target data
    combined_test_imgs = x_target[:num_test]
    combined_test_labels = y_train[:num_test]
    combined_test_domain = np.tile([1., 0.], [num_test, 1])
    dann_embedding = dann_vis.predict(combined_test_imgs)
    np.save('dann_embedding_for_unsupervised', dann_embedding)
    # np.save('shuffled_labels', combined_test_labels)
    dann_tsne = TSNE(perplexity=70, n_components=2, init='pca', n_iter=5000)
    tsne = dann_tsne.fit_transform(dann_embedding)
    plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'DANN')
    plt.savefig('DANN_tSNE_testi')
    # ----------------------------------
    
    # src_embedding = src_vis.predict([combined_test_imgs])
    # src_tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=3000)
    # tsne = src_tsne.fit_transform(src_embedding)

    # plot_embedding(tsne, combined_test_labels.argmax(1),
                   # combined_test_domain.argmax(1), 'Source only')
    # plt.savefig('SOURCE_tSNE')
 
if make_movie:
    import cv2
    import os
    cap = cv2.VideoCapture(0)
    image_folder = 'img/'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    #    cv2.destroyAllWindows()
    cap.release()
    video.release()
    cv2.waitKey()
    cv2.destroyAllWindows()

            


