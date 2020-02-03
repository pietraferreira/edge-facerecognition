#from tensorflow.contrib import lite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model_file( 'model/facenet_keras.h5')
tfmodel = converter.convert()
open ("facenet_keras.tflite" , "wb") .write(tfmodel)
