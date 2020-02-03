from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

#load model
model = VGG16()
#load image
image = load_img('car.jpg', target_size=(224, 224))
#convert the image from pixels to an array
image = img_to_array(image)
#reshape
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#prepare image for the model
image = preprocess_input(image)
#predict the probability across all output classes
yhat = model.predict(image)
#convert the probabilities to class labels
label = decode_predictions(yhat)
#retrieve most likely result
label = label[0][0]
#print classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
