from keras.models import Sequential
from keras.layers import Dense, Dropout, Input,GaussianDropout, GaussianNoise
# from keras.layers.
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Deconv2D, Conv2DTranspose,AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, adam, adadelta
from keras.datasets import mnist
from keras.models import Model
import numpy as np
from PIL import Image
import argparse
import os
import math
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as misc
from keras.callbacks import TensorBoard
import random


os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def generator_model5():
    input = Input([5,5,3])
    x  = Conv2D(1,(1,1),padding='same')(input)
    x = Flatten()(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dense(2048, activation='tanh')(x)
    x = Dense(64*25*25, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Reshape((25,25,64))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(5,5),padding='same',activation='tanh')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (5, 5), padding='same', activation='tanh')(x)
    model = Model(inputs=input, outputs=x)
    return model

def generator_model50():
    input = Input([50,50,3])
    x  = Conv2D(1,(1,1),padding='same')(input)
    x = Flatten()(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dense(2048, activation='tanh')(x)
    x = Dense(64*25*25, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Reshape((25,25,64))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(5,5),padding='same',activation='tanh')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (5, 5), padding='same', activation='tanh')(x)
    model = Model(inputs=input, outputs=x)
    return model

def generator_model500():
    input = Input([500,500,3])
    x  = Conv2D(1,(1,1),padding='same')(input)
    x = Flatten()(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dense(2048, activation='tanh')(x)
    x = Dense(64*25*25, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Reshape((25,25,64))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(5,5),padding='same',activation='tanh')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (5, 5), padding='same', activation='tanh')(x)
    model = Model(inputs=input, outputs=x)
    return model

def discriminator_model5():
    input = Input([5,5,3])
    x = Conv2D(64,(5,5),padding='same', activation='tanh')(input)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128,(5,5), activation='tanh')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    d = Dense(1, activation='sigmoid')(x)
    cls = Dense(51, activation='softmax')(x)
    model = Model(inputs=input, outputs=[d, cls])
    return model

def discriminator_model50():
    input = Input([50,50,3])
    x = Conv2D(64,(5,5),padding='same', activation='tanh')(input)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128,(5,5), activation='tanh')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    d = Dense(1, activation='sigmoid')(x)
    cls = Dense(51, activation='softmax')(x)
    model = Model(inputs=input, outputs=[d, cls])
    return model

def discriminator_model500():
    input = Input([500,500,3])
    x = Conv2D(64,(5,5),padding='same', activation='tanh')(input)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128,(5,5), activation='tanh')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    d = Dense(1, activation='sigmoid')(x)
    cls = Dense(51, activation='softmax')(x)
    model = Model(inputs=input, outputs=[d, cls])
    return model

def generator_containing_discriminator(g, d):
    d.trainable = False
    model = Model(inputs=g.inputs, outputs=d(g.outputs))
    return model

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=5, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    # print window
    K1 = 0.01
    K2 = 0.03
    L = 255  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def structural_loss(y_true,y_pred):
    img1 = tf.image.rgb_to_grayscale(y_true)
    img2 = tf.image.rgb_to_grayscale(y_pred)
    print img1
    ssim_loss = 1-tf_ms_ssim(img1,img2)
    # bc_loss = K.binary_crossentropy(y_true,y_pred)
    mse_loss = K.mean(K.square(y_true - y_pred))
    st_loss = K.abs((ssim_loss + mse_loss)/2)
    return st_loss