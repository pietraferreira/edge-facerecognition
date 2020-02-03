from tensorflow.python.keras import models, optimizers, losses, activations, callbacks
from tensorflow.python.keras.layers import *
import tensorflow.python.keras.backend as K
from PIL import Image
import tensorflow as tf
import time
import os
import numpy as np

class Recogniser(object):
    def __init__(self):
        #display errors
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.__DIMEN = 128

        input_shape = ((self.__DIMEN**2) * 3,)
        conv_shape = (self.__DIMEN, self.__DIMEN, 3)
        kernel_size1 = (4,4)
        kernel_size2 = (3,3)
        pool_size1 = (3,3)
        pool_size2 = (2,2)
        strides = 1

        seq_conv_model = [
            Reshape(input_shape=input_shape, target_shape=conv_shape),
            Conv2D(32, kernel_size=kernel_size1, strides=strides, activation=activations.leaky_relu),
            Conv2D(32, kernel_size=kernel_size1, strides=strides, activation=activations.leaky_relu),
            MaxPooling2D(pool_size=pool_size1, strides=strides),

            Conv2D(64, kernel_size=kernel_size2, strides=strides, activation=activations.leaky_relu),
            Conv2D(64, kernel_size=kernel_size2, strides=strides, activation=activations.leaky_relu),
            MaxPooling2D(pool_size=pool_size2, strides=strides),

            Flatten(),

            Dense(64, activation=activations.sigmoid)
        ]

        seq_model = tf.keras.Sequential(seq_conv_model)

        input_x1 = Input(shape=input_shape)
        input_x2 = Input(shape=input_shape)

        output_x1 = seq_model(input_x1)
        output_x2 = seq_model(input_x2)

        distance_euclid = Lambda(lambda tensors : K.abs(tensors[0] - tensors[1]))([output_x1, output_x2])
        outputs = Dense(1, activation=activations.sigmoid)(distance_euclid)
        self.__model = models.Model([input_x1, input_x2], outputs)

        self.__model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.0001))

    def fit(self, X, y, hyperparameters):
        initial_time = time.time()
        self.__model.fit(X, y, batch_size=hyperparameters['batch_size'],
                               epochs=hyperparameters['epochs'],
                               callbacks=hyperparameters['callbacks'],
                               validation_data=hyperparameters['val_data']
                        )
        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        self.__model.summary()
        print("Elapsed time acquired for {} epoch(s) -> {} {}".format(hyperparameters['epochs'], eta, time_unit))

        def prepare_images_from_dir(self, dir_path, flatten=True):
            images = list()
            images_names = os.listdir(dir_path)
            for imageName in images_names:
                image = Image.open(dir_path + imageName)
                resize_image = image.resize((self.__DIMEN, self.__DIMEN))
                array = list()
                for x in range(self.__DIMEN):
                    sub_array = list()
                    for y in range(self.__DIMEN):
                        sub_array.append(resize_image.load()[x,y])

            if flatten:
                images = np.array(images)
                return image.reshape((images.shape[0], self.__DIMEN**2 * 3)).astype(np.float32)
            else:
                return np.array(images)

        def evaluate(self, test_X, test_y):
            return self.__model.evaluate(test_X, test_y)

        def predict(self, X):
            predictions = self.__model.predict(X)
            return predictions

        def summary(self):
            self.__model.summary()

        def save_model(self, file_path):
            self.__model.save(file_path)

        def load_model(self, file_path):
            self.__model = models.load_model(file_path)
