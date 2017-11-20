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
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Model
#from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.datasets import mnist
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix

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
                                 

class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = SGD()

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Convolution1D(nb_filters, nb_conv,
                            border_mode='valid',
                            activation='relu')(model_input)
        # net = Convolution1D(nb_filters, nb_conv,
                            # activation='relu')(net)
        net = MaxPooling1D(pool_size=nb_pool)(net)
        net = Dropout(0.5)(net)
        net = Flatten()(net)
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(128, activation='relu')(model_input)
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
        branch = Dropout(0.1)(branch)
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
                      optimizer=self.opt, metrics=['accuracy'])
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
batch_size = 256
batch_size_test = 256
nb_epoch = 1000
nb_classes = 2                  # number of classes
input_dim = 924                  # input size
nb_filters = 32                 # number of filters in Conv1D
nb_pool = 2                     # pool size of Conv1D
nb_conv = 3                     # Size of receptive field
_TRAIN = K.variable(1, dtype='uint8')
loading = True
train_source_only = False
train_dann = True
evaluate_model = True
analyze_predictions = True
unsupervised = True
L = 12

# ----------------------------------------------------------------
# Prepare Data
# We dont have to shuffle, we do it in train_generator
len_x_train = 10000
len_x_target = 10000
rescaler = 0

folder = 'MBL_data/'

train_weight_path = folder + 'weights_MBL_'
test_weight_path = folder + 'weights200.npy'
unsuperivised_weight_path = folder + 'unsupervised_MBL_'+str(len_x_train)+'.npy'
pred_target_path = folder+'predictions_on_target_MBL_'+str(len_x_train)
pred_source_path = folder+'predictions_on_train_MBL_'+str(len_x_train)



if loading:
    path_source = folder+'Training_Set_09_to_50_2' + '_N' + str(L) + '.h5'
    path_target = folder+'TARGET_Set_09_to_50_2' + '_N' + str(L) + '.h5'
    #path_target = 'TARGET_ABSOLUTE_OBC_disorder_w1_1.0_w2_05w1_not_shuffled_full_N64.h5'
    x_train, y_train = train_loader(0, len_x_train, path_source)
    y_train = np.array(y_train)
    x_target, y_target = train_loader(0, len_x_target, path_target)
    del y_target
    source_index_arr = np.arange(x_train.shape[0])
    target_index_arr = np.arange(x_target.shape[0])
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

    src_model.fit(x_train, y_train, batch_size=64, nb_epoch=10, verbose=1, shuffle = 'batch')
    # shuffle = 'batch' is used for HDF5 files
    print('Evaluating target samples on source-only model')
    print('Accuracy: ', src_model.evaluate(x_target, y_target)[1])
# -------------------------------------------------------------------


batches_per_epoch = len_x_train / batch_size
num_steps = nb_epoch * batches_per_epoch
j = 0            
if train_dann:   
    print('Training DANN model')
    for i in range(nb_epoch):
        
        # batches = make_batches(X_train.shape[0], batch_size / 2)
        # target_batches = make_batches(XT_train.shape[0], batch_size / 2)

        src_gen = batch_generator(len_x_train, source_index_arr, batch_size // 2, x_train, y_train)
        target_gen = batch_generator(len_x_target, target_index_arr, batch_size // 2, x_target, None)
            
        losses = list()
        acc = list()

        print('Epoch ', i)
        np.save(folder+'weights'+str(i), dann_model.get_weights()) # save weights after each epoch
        #        if i == 0:
        #            weight = np.load('weights_w1_02_50_epochs.npy') # reload weights
        #            dann_model.set_weights(weight)
        #        else:
        #            np.save('weights', dann_model.get_weights())
        np.save(train_weight_path, dann_model.get_weights())
        # dann_model.set_weights(weight)
        device = "gpu"
        with tf.device("/" + device + ":0"):
            for (xb, yb) in src_gen:
                # Update learning rate and gradient multiplier as described in
                # the paper.
                p = float(j) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p)**0.75
                builder.grl.l = l
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
                xb = np.vstack([xb, xt])

                print(j)
                #print('xb', len(xb), 'yb', len(yb), 'domain', len(domain_labels))
                metrics = dann_model.train_on_batch({'main_input': xb},
                                                    {'classifier_output': yb,
                                                    'domain_output': domain_labels}, check_batch_dim=False)
                j += 1


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
    for start in range(0, len_x_train, batch_size_test):
        print(start)
        end = min(start + batch_size_test, len_x_train)
        out2 = dann_model.predict_on_batch(x_target[start:end])
        out1 = dann_model.predict_on_batch(x_train[start:end])
        #print(out2[1])
        list_target.append(out2)
        list_train.append(out1)
    #print(list)
    np.save( pred_target_path, list_target)
    np.save( pred_source_path, list_train)
    #print(out1)
    #print(out2)
    out = np.argmax(out2[1], axis=1)
    actual = np.argmax(y_train[0:batch_size // 2], axis=1)
    acc = float(np.sum((out == actual))) / float(len(out))
    print('Accuracy: ', acc)
    print('Visualizing output of domain invariant features')


if analyze_predictions:
    list_target = np.load( pred_target_path+'.npy') #40001 states
    list_train = np.load( pred_source_path+'.npy')  #40001 states
    train_phase = []
    target_phase = []
    for j in range(0, len_x_train // batch_size_test):
        for i in range(batch_size_test):
            train_phase.append(list_train[j][1][i][0])
            target_phase.append(list_target[j][1][i][0])
      
    plt.clf()
    plt.plot(np.linspace(0,1,len(train_phase)), train_phase)
    plt.savefig(folder+'train')
    #a, b = zip(*sorted(zip(x_axis[:len(target_phase)], target_phase)))
    # sort shuffled data for plot
    plt.clf()
    plt.plot(np.linspace(0,1,len(train_phase)), target_phase)
    plt.savefig(folder+'target')
   

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
    np.save('shuffled_labels', combined_test_labels)
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

            


