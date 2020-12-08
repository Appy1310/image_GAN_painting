''' 
Class defining a CycleGAN model for image to image translation
'''

# import random
# import os
# #from os import listdir
# from random import random
# import numpy as np
# from numpy import load, zeros, ones, asarray
# from numpy.random import randint
from tensorflow import keras
import tensorflow as tf

#import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import Callback
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#from matplotlib import pyplot

from utils import  generate_real_samples, generate_fake_samples
from utils import save_models, summarize_performance, update_image_pool

# Build the CycleGAN class
''' It has the following  methods:
discriminator:
generator:
'''
# # generator: A -> B
# g_model_AtoB = model.define_generator(image_shape)
# # generator: B -> A
# g_model_BtoA = model.define_generator(image_shape)
# # discriminator: A -> [real/fake]
# d_model_A = model.define_discriminator(image_shape)
# # discriminator: B -> [real/fake]
# d_model_B = model.define_discriminator(image_shape)
# # composite: A -> B -> [real/fake, A]
# c_model_AtoB = model.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# # composite: B -> A -> [real/fake, B]
# c_model_BtoA = model.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

class CycleGAN(tf.keras.Model):

    def __init__(self, image_shape):
        super(CycleGAN, self).__init__()
        self.image_shape = image_shape
        # generator: A -> B
        self.g_model_AtoB = self.define_generator()
        # generator: B -> A
        self.g_model_BtoA = self.define_generator()
        # discriminator: A -> [real/fake]
        self.d_model_A = self.define_discriminator()
        # discriminator: B -> [real/fake]
        self.d_model_B = self.define_discriminator()
        # composite: A -> B -> [real/fake, A]
        self.c_model_AtoB = self.define_composite_model(self.g_model_AtoB, 
                                          self.d_model_B, self.g_model_BtoA, image_shape)
        # composite: B -> A -> [real/fake, B]
        self.c_model_BtoA = self.define_composite_model(self.g_model_BtoA, 
                                          self.d_model_A, self.g_model_AtoB, image_shape)

    def define_discriminator(self):
        ''' defines the discriminator function'''
        # weight initialization
        init = keras.initializers.RandomNormal(stddev=0.02)
        # source image input
        input_image = keras.Input(shape=self.image_shape)
        # C64
        d_layer = keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_image)
        d_layer = keras.layers.LeakyReLU(alpha=0.2)(d_layer)
        # C128
        d_layer = keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d_layer)
        d_layer = InstanceNormalization(axis=-1)(d_layer)
        d_layer = keras.layers.LeakyReLU(alpha=0.2)(d_layer)
        # C256
        d_layer = keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d_layer)
        d_layer = InstanceNormalization(axis=-1)(d_layer)
        d_layer = keras.layers.LeakyReLU(alpha=0.2)(d_layer)
        # C512
        d_layer = keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d_layer)
        d_layer = InstanceNormalization(axis=-1)(d_layer)
        d_layer = keras.layers.LeakyReLU(alpha=0.2)(d_layer)
        # second last output layer
        d_layer = keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d_layer)
        d_layer = InstanceNormalization(axis=-1)(d_layer)
        d_layer = keras.layers.LeakyReLU(alpha=0.2)(d_layer)
        # patch output
        patch_out = keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d_layer)
        # define model
        model = keras.Model(input_image, patch_out)
        # compile model
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        return model

    # generator of a resnet block
    def resnet_block(self, n_filters, input_layer):
        ''' generates a resnet_block'''
        # weight initialization
        init = keras.initializers.RandomNormal(stddev=0.02)
        # first layer convolutional layer
        g_layer = keras.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        g_layer = keras.layers.Activation('relu')(g_layer)
        # second convolutional layer
        g_layer = keras.layers.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g_layer)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        # concatenate merge channel-wise with input layer
        g_layer = keras.layers.Concatenate()([g_layer, input_layer])
        return g_layer   

    # define the standalone generator model
    def define_generator(self, n_resnet=9):
        ''' defines a generator function'''
        # weight initialization
        init = keras.initializers.RandomNormal(stddev=0.02)
        # image input
        input_image = keras.Input(shape= self.image_shape)
        # c7s1-64
        g_layer = keras.layers.Conv2D(64, (7,7), padding='same',
                                       kernel_initializer=init)(input_image)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        g_layer = keras.layers.Activation('relu')(g_layer)
        # d128
        g_layer = keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same',
                                        kernel_initializer=init)(g_layer)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        g_layer = keras.layers.Activation('relu')(g_layer)
        # d256
        g_layer = keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same',
                                        kernel_initializer=init)(g_layer)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        g_layer = keras.layers.Activation('relu')(g_layer)
        # R256
        for _ in range(n_resnet):
            #print('running through loop!')
            g_layer = self.resnet_block(256, g_layer)
        # u128
        g_layer = keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), 
                              padding='same', kernel_initializer=init)(g_layer)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        g_layer = keras.layers.Activation('relu')(g_layer)
        # u64
        g_layer = keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), 
                              padding='same', kernel_initializer=init)(g_layer)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        g_layer = keras.layers.Activation('relu')(g_layer)
        # c7s1-3
        g_layer = keras.layers.Conv2D(3, (7,7), padding='same', 
                                              kernel_initializer=init)(g_layer)
        g_layer = InstanceNormalization(axis=-1)(g_layer)
        out_image = keras.layers.Activation('tanh')(g_layer)
        # define model
        model = keras.Model(input_image, out_image)
        return model    


    # define a composite model for updating generators by adversarial and cycle loss
    def define_composite_model(self, g_model_1, d_model, g_model_2, image_shape):
        '''define a composite model for updating generators
                                 by adversarial and cycle loss'''
        # ensure the model we're updating is trainable
        g_model_1.trainable = True
        # mark discriminator as not trainable
        d_model.trainable = False
        # mark other generator model as not trainable
        g_model_2.trainable = False
        # discriminator element
        input_gen = keras.Input(shape=image_shape)
        gen1_out = g_model_1(input_gen)
        output_d = d_model(gen1_out)
        # identity element
        input_id = keras.Input(shape=image_shape)
        output_id = g_model_1(input_id)
        # forward cycle
        output_f = g_model_2(gen1_out)
        # backward cycle
        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)
        # define model graph
        model = keras.Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        # define optimization algorithm configuration
        opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
        return model
    
    # train cyclegan models
    def train(self, d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, 
                                 c_model_AtoB, c_model_BtoA, dataset):
        '''  tarining step for cyclegan models'''
        # define properties of the training run
        n_epochs, n_batch, = 50, 1
        # determine the output square shape of the discriminator
        n_patch = d_model_A.output_shape[1]
        # unpack dataset
        trainA, trainB = dataset
        # prepare image pool for fakes
        poolA, poolB = list(), list()
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
            X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
            # update fakes from pool
            X_fakeA = update_image_pool(poolA, X_fakeA)
            X_fakeB = update_image_pool(poolB, X_fakeB)
            # update generator B->A via adversarial and cycle los
            g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA],
                                            [y_realA, X_realA, X_realB, X_realA])
            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], 
                                            [y_realB, X_realB, X_realA, X_realB])
            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            # summarize performance
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
            # evaluate the model performance every so often
            if (i+1) % (bat_per_epo * 1) == 0:
                # plot A->B translation
                summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
                # plot B->A translation
                summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
            if (i+1) % (bat_per_epo * 5) == 0:
                # save the models
                save_models(i, g_model_AtoB, g_model_BtoA)
   