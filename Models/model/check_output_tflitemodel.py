import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="working_model.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details()[0]['shape'])
print(interpreter.get_input_details()[0]['dtype'])

print(interpreter.get_output_details()[0]['shape'])
print(interpreter.get_output_details()[0]['dtype'])
