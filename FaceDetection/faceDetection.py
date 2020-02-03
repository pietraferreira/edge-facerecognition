import matplotlib
matplotlib.use("Agg")
import numpy as np
import cv2
import matplotlib.pyplot as plt

#load image
sample_image = cv2.imread('barack_obama.jpg')

#convert to greyscale
image_to_gray = cv2.cvtColor(sample_image, cv2.COLORBGR2GRAY)

#openCV loads images in BGR format not RGB
#therefore to be able to see the image in colour
#we need to convert it
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')

#here ill be performing face detection
faces = haar_cascade_face.detectMultiScale(image_to_gray, scaleFactor = 1.2, minNeighbors = 5)

print('Faces found: ', len(faces))

#now to draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#convert to RGB
plt.imshow(convertToRGB(image))
