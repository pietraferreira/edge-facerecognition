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

num_classes = 1872
epochs = 100
root = '/Users/pietraferreira/dataset/'
labels = []
data = np.empty((0,512), np.float32)

for index, file in enumerate(os.listdir(root)):
    for embeddings in os.listdir(os.path.join(root, file)):
        if 'embedding' in embeddings:
            #print(data)
            labels.append(index)
            data = np.append(data, np.array([np.loadtxt(os.path.join(root, file, embeddings))]), axis=0)

one_hot_labels = to_categorical(labels, num_classes=num_classes)
print(labels)

def create_keras_model():
    model = Sequential()
    model.add(keras.layers.BatchNormalization(input_shape=(512,)))
    model.add(Dense(num_classes, activation='softmax', bias_initializer='zeros'))
    #model.add(Dense(3, activation='softmax'))
    return model

train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)

with train_graph.as_default():
    model = create_keras_model()
    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())
    #compile model
    sgdop = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgdop, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data, one_hot_labels, epochs=epochs, batch_size=32, shuffle=True)
    
    saver = tf.train.Saver()
    saver.save(train_sess, 'checkpoints')

# model.save(model_name + ".h5")
# model_name = "new_model"

#print(model.summary())
# print(model.input.op.name)
# print(model.input.op.name)
# print(model.output.op.name)
# print("Saved model to disk!")

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    eval_model = create_keras_model()
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    #freeze the graph
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(eval_sess, eval_graph_def, [eval_model.output.op.name])

    #WB stands for: writing truncating the file first and in binary mode
    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

print(model.summary())

