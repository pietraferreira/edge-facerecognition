import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
import os
from keras import backend as K
from keras.models import save_model
import tensorflow as tf

#def my_init(shape, dtype=None):
    #return 5e-2 * np.random.randn(shape[0], shape[1]).astype(np.float32)


#initializer = keras.initializers.RandomNormal(mean=0.0, stddev=5e-2, seed=None)

model = Sequential()

model.add(keras.layers.BatchNormalization(input_shape=(512,)))
model.add(Dense(23, activation='softmax', bias_initializer='zeros'))
#model.add(Dense(3, activation='softmax'))
sgdop = keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgdop, loss='categorical_crossentropy', metrics=['accuracy'])

root = '/Users/pietraferreira/dataset/'
#data = np.empty((0,512), float)
data = np.empty((0,512), np.float32)
labels = []

for index, file in enumerate(os.listdir(root)):
    for embeddings in os.listdir(os.path.join(root, file)):
        if 'embedding' in embeddings:
            #print(data)
            labels.append(index)
            data = np.append(data, np.array([np.loadtxt(os.path.join(root, file, embeddings))]), axis=0)

one_hot_labels = to_categorical(labels, num_classes=23)

#print(model.summary())
print(model.input.op.name)

model.fit(data, one_hot_labels, epochs=1000, batch_size=32, shuffle=True)

print(model.input.op.name)
print(model.output.op.name)
#saver = tf.train.Saver()
#saver.save(K.get_session(), 'keras_model.ckpt')

#model.save("yas_model.h5")
#print("Saved model to disk!")
