import random
import os
import sys
import urllib.request
import shutil #from os import listdir
from random import random
import numpy as np
from numpy import load, zeros, ones, asarray
from numpy.random import randint
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

import matplotlib.pyplot as plt
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot


# # load the models
# cust = {'InstanceNormalization': InstanceNormalization}
# #from vangogh to photo
# Download the file from `url` and save it locally under `file_name`:
# vangogh_url = 'https://drive.google.com/file/d/1LdMnbhppbR6KMOvPNF2Hc3gxwmqAIOw6/view?usp=sharing'
# with open(file_name, "wb") as file:
#     # get request
#     response = get(vangogh_url)
#     # write to file
#     file.write(response.content)
# #vangogh_model = requests.get(vangogh_url, allow_redirects=True)
# print(type(file))
# model = load_model('vangogh_model')
# print('it is loaded!')
# model_vangogh = load_model('/home/aprameyo/Image_GAN/pretrained_models/model_photo_to_vangogh.h5', cust , compile=False)
# model_vangogh.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
# #from photo to vangogh
# model_monet = load_model('/home/aprameyo/Image_GAN/pretrained_models/model_photo_to_monet.h5', cust, compile=False)
# model_monet.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))





def load_image(filename, size=(256,256)):
    # load and resize the image
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # transform in a sample
    pixels = expand_dims(pixels, 0)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    return pixels

def paint_generator(model, image, name):
    # translate image
    image_tar = model.predict(image)
    # scale from [-1,1] to [0,1]
    image_tar = (image_tar + 1) / 2.0
    # plot the translated image
    plt.imshow(image_tar[0])
    plt.axis('off')
    plt.savefig(f'./static/img/{name}')
    plt.show()    

print('It is working!')    