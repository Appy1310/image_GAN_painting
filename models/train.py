import random
import os
import sys
#from os import listdir
from random import random
import numpy as np
from numpy import load, zeros, ones, asarray
from numpy.random import randint
from tensorflow import keras
import matplotlib.pyplot as plt
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot


from CycleGAN import CycleGAN
from utils import  generate_real_samples, generate_fake_samples, load_real_samples
from utils import save_models, summarize_performance, update_image_pool



# sample data with proper path
sample_data = sys.argv[1]

# load image data
dataset = load_real_samples(sample_data)
print('Loaded', dataset[0].shape, dataset[1].shape)


# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)

model = CycleGAN(image_shape)
print('model loading succesful!')

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



# train models
#model.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)