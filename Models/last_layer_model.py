import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
import os
from keras import backend as K
from keras.models import save_model
import tensorflow as tf

train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)

#def my_init(shape, dtype=None):
    #return 5e-2 * np.random.randn(shape[0], shape[1]).astype(np.float32)
num_classes = 1872
epochs = 300

#initializer = keras.initializers.RandomNormal(mean=0.0, stddev=5e-2, seed=None)

model = Sequential()

model.add(keras.layers.BatchNormalization(input_shape=(512,)))
model.add(Dense(num_classes, activation='softmax', bias_initializer='zeros'))
#model.add(Dense(3, activation='softmax'))
sgdop = keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgdop, loss='categorical_crossentropy', metrics=['accuracy'])

with train_graph.as_default():
    train_model = model
    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())
    #compile model
    model.compile(optimizer=sgdop, loss='categorical_crossentropy', metrics=['accuracy']) 


model_name = "new_model"
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

one_hot_labels = to_categorical(labels, num_classes=num_classes)

#print(model.summary())
print(model.input.op.name)

model.fit(data, one_hot_labels, epochs=epochs, batch_size=32, shuffle=True)

print(model.input.op.name)
print(model.output.op.name)
saver = tf.train.Saver()
saver.save(train_sess, 'checkpoints')

model.save(model_name + ".h5")
print("Saved model to disk!")

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)
