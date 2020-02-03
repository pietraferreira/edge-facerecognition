import tensorflow as tf

#Enter the model's directory (format = .h5)
converter = tf.lite.TFLiteConverter.from_keras_model_file('updated_model.h5')

model = converter.convert()

file = open('updated_model.tflite', 'wb')
file.write(model)
