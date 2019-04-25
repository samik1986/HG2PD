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

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    # channel = 3
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    # print shape[2]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    # image1 = np.zeros((height * shape[0], width * shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    # c_image = np.zeros((height*shape[0], 2*width*shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    for index, rgb in enumerate(generated_images):
        # r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # img1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
        img = rgb
        # print img.shape
        # print img[:,:,0]
        i = int(index/width)
        # print i
        j = index % width
        # print j
        # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1],0] = img1[:,:]
        # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],0] = img[:, :,0]
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 1] = img[:, :, 1]
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 2] = img[:, :, 2]
    # c_image[0:height*shape[0], 0:width*shape[1], 0:shape[2]] = image
    # c_image[0:height*shape[0], width*shape[1]:2*width*shape[1], 0:shape[2]] = image1
    return image

def save_images(generated_images,epoch,batch):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    # channel = 3
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    # print shape[2]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    # image1 = np.zeros((height * shape[0], width * shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    # c_image = np.zeros((height*shape[0], 2*width*shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    for index, rgb in enumerate(generated_images):
        # r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # img1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
        img = rgb
        img = img * 127.5 + 127.5
        Image.fromarray(img.astype(np.uint8)).save("Images1/" + str(epoch) +"_" + str(index) + "_" + str(batch) + ".png")
        # print img.shape
        # print img[:,:,0]
        # i = int(index/width)
        # # print i
        # j = index % width
        # # print j
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1],0] = img1[:,:]
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],0] = img[:, :,0]
        # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 1] = img[:, :, 1]
        # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 2] = img[:, :, 2]
    # c_image[0:height*shape[0], 0:width*shape[1], 0:shape[2]] = image
    # c_image[0:height*shape[0], width*shape[1]:2*width*shape[1], 0:shape[2]] = image1
    return image

def save_images_or(generated_images,epoch):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    # channel = 3
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    # print shape[2]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    # image1 = np.zeros((height * shape[0], width * shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    # c_image = np.zeros((height*shape[0], 2*width*shape[1], shape[2]),
    #                  dtype=generated_images.dtype)
    for index, rgb in enumerate(generated_images):
        # r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # img1 = 0.2989 * r + 0.5870 * g + 0.1140 * b
        img = rgb
        img = img * 127.5 + 127.5
        Image.fromarray(img.astype(np.uint8)).save("Images2/" + str(epoch) +"_" + str(index) + ".png")
        # print img.shape
        # print img[:,:,0]
        # i = int(index/width)
        # # print i
        # j = index % width
        # # print j
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1],0] = img1[:,:]
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
        # image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],0] = img[:, :,0]
        # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 1] = img[:, :, 1]
        # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 2] = img[:, :, 2]
    # c_image[0:height*shape[0], 0:width*shape[1], 0:shape[2]] = image
    # c_image[0:height*shape[0], width*shape[1]:2*width*shape[1], 0:shape[2]] = image1
    return image

# def combine_images_or(generated_images):
#     num = generated_images.shape[0]
#     width = int(math.sqrt(num))
#     # channel = 3
#     height = int(math.ceil(float(num)/width))
#     shape = generated_images.shape[1:3]
#     # print shape[2]
#     image = np.zeros((height*shape[0], width*shape[1]),
#                      dtype=generated_images.dtype)
#     # image1 = np.zeros((height * shape[0], width * shape[1], shape[2]),
#     #                  dtype=generated_images.dtype)
#     # c_image = np.zeros((height*shape[0], 2*width*shape[1], shape[2]),
#     #                  dtype=generated_images.dtype)
#     for index, rgb in enumerate(generated_images):
#         r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#         img = 0.2989 * r + 0.5870 * g + 0.1140 * b
#         # img = rgb
#         # print img.shape
#         # print img[:,:,0]
#         i = int(index/width)
#         # print i
#         j = index % width
#         # print j
#         # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1],0] = img1[:,:]
#         # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
#         # image1[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 0] = img1[:,:]
#         image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :]
#         # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 1] = img[:, :, 1]
#         # image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], 2] = img[:, :, 2]
#     # c_image[0:height*shape[0], 0:width*shape[1], 0:shape[2]] = image
#     # c_image[0:height*shape[0], width*shape[1]:2*width*shape[1], 0:shape[2]] = image1
#     return image


def train(BATCH_SIZE):
    d = discriminator_model()
    print d.summary()
    # d.load_weights('Models/D/discriminator1594')
    g = generator_model()
    print g.summary()
    X_tr = np.load('trainImagesIITM_n.npy')
    L_tr = np.load('trainLabelIITM_1hot.npy')
    # L_tr = np.arange(L_tr,)
    Y_tr = np.load('testImagesIITM_n.npy')
    # plt.imshow(Y_train[0])
    # plt.show()
    # print X_train.shape
    X_tr = (X_tr.astype(np.float32)-127.5) / 127.5
    Y_tr = (Y_tr.astype(np.float32)-127.5) / 127.5
    # print len(Y_tr)
    result = []
    X_trn = []
    Y_trn = []
    L_trn = []
    for x in range(0, len(Y_tr)):
        num = random.randint(0, len(Y_tr)-1)
        while num in result:
            num = random.randint(0, len(Y_tr)-1)
        result.append(num)
    # print len(result)

    # X_train[:,:,:,:] = X_tr[ for x in result]

    for i in result:
        X_im = X_tr[i]
        X_trn.append(X_im)
        L_im = L_tr[i]
        L_trn.append(L_im)
        Y_im = Y_tr[i]
        Y_trn.append(Y_im)
    X_train = np.asarray(X_trn)
    Y_train = np.asarray(Y_trn)
    L_train = np.asarray(L_trn)

    print L_train.shape
    # Xtra
    # X0_train = X_train[:, :, :, 0]
    # Y0_train = Y_train[:, :, :, 0]
    # X1_train = X_train[:, :, :, 1]
    # Y1_train = Y_train[:, :, :, 1]
    # X2_train = X_train[:, :, :, 2]
    # Y2_train = Y_train[:, :, :, 2]
    # # X_train = X0_train
    # X_train = np.concatenate((X0_train,X1_train,X2_train),axis=0)
    # Y_train = np.concatenate((Y0_train, Y1_train, Y2_train),axis=0)
    # X_train = np.expand_dims(X_train,axis=-1)
    # Y_train = np.expand_dims(Y_train, axis=-1)
    # Y_train = K.flatten(Y_train)
    # X_train = np.append(X2_train)
    # print X_train.shape
    # X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    g_loss = 0.
    d_loss = 0.00001
    # alpha_loss = 0.
    # fig = plt.figure()
    # plt.axis([0, 100000000, 0, 1])
    # i = 0
    s_x = []
    O_X = []
    N_Y = []
    g_x = []
    d_x = []
    # g.load_weights('Models/G/generator1594')
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)
    # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-5)
    # adadelta = keras.optimizers.adadelta(lr=0.0001, decay=1e-5)
    g.compile(loss=structural_loss, optimizer=g_optim)
    d_on_g.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=g_optim)
    d.trainable = True
    d.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=d_optim)
    for epoch in range(10000):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        # count = 0
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            # noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            noise = Y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            noise_label = L_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            # print noise_label.shape
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            # print generated_images.shape
            if index % 60 == epoch % 60:
                image = combine_images(generated_images)
                image_or =combine_images(image_batch)
                # p = save_images(generated_images, epoch, index)
                # q = save_images_or(image_batch, epoch)
                # for i in range(BATCH_SIZE):
                #     O_X.append(image_batch[i,])
                #     N_Y.append(generated_images[i,])
                # np.save('original_images.npy',O_X)
                # np.save('noisy_gen_images.npy',N_Y)
                image = image * 127.5 + 127.5
                image_or = image_or * 127.5 + 127.5
                # image_gray = image_gray * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save("Images/"+
                    str(epoch) + "_" + str(index) + ".png")
                Image.fromarray(image_or.astype(np.uint8)).save("Images1/" +
                    str(epoch) + "_" + str(index) + "_or.png")
                # if count<60:
                #     count = count+1
                # else:
                #     count = 0
                # Image.fromarray(image_gray.astype(np.uint8)).save("Images/" +
                #     str(epoch) + "_" + str(index) + "_gray.png")
            if epoch > 6:
                g_trend = sum(g_x[epoch-6:epoch][0])/5.
                d_trend = sum(d_x[epoch-6:epoch][0])/5.
                diff = g_trend - d_trend
                ratio = g_loss[0]/d_loss[0]

                if diff > 1 or ratio >1000:
                    g.trainable = False
                if diff < -1 or ratio < 0.001:
                    d.trainable = False

            X = image_batch
            y = [1] * BATCH_SIZE
            y = np.asarray(y)
            d_loss = d.train_on_batch(X, [y, noise_label])
            X = generated_images
            y = [0] * BATCH_SIZE
            y = np.asarray(y)
            d_loss = d.train_on_batch(X, [y, noise_label])
            # print("batch %d d_loss : %f" % (index, d_loss))
            # noise = Y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]


            d.trainable = False
            alpha_loss = g.train_on_batch(noise,image_batch)
            y = [1] * BATCH_SIZE
            y = np.asarray(y)
            g_loss = d_on_g.train_on_batch(noise, [y, noise_label])
            # print d_on_g.summary()
            d.trainable = True
            g.trainable = True
                # print d_on_g.summary()
            print("epoch %d batch %d d_loss : %f %f g_loss : %f %f s_loss : %f" % (epoch, index,
                                                                                   d_loss[0], d_loss[1],
                                                                                   g_loss[0], g_loss[1],
                                                                                   alpha_loss))
            # plt.scatter(i, g_loss)
            # i += 1
            # plt.show()
            if epoch % 200 == 99:
                g.save_weights('Models/G/generator1%d'%epoch, True)
                d.save_weights('Models/D/discriminator1%d'%epoch, True)
                np.save('Models/generator1.npy',g_x,True)
                np.save('Models/discriminator1.npy', d_x, True)
                np.save('Models/structural1.npy', s_x, True)

        g_x.append(g_loss)
        d_x.append(d_loss)
        s_x.append(alpha_loss)

def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='structural_loss', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")

train(20)

# generate(1,nice=True)

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str)
#     parser.add_argument("--batch_size", type=int, default=30)
#     parser.add_argument("--nice", dest="nice", action="store_true")
#     parser.set_defaults(nice=False)
#     args = parser.parse_args()
#     return args
#
# if __name__ == "__main__":
#     args = get_args()
#     if args.mode == "train":
#         train(BATCH_SIZE=args.batch_size)
#     elif args.mode == "generate":
#         generate(BATCH_SIZE=args.batch_size, nice=args.nice)