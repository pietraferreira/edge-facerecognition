from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load face detector
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#load embedder
print("[INFO] loading face recogniser...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

#grab images and perform initialisations
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

#list of extracted facial embeddings and corresponding labels
knownEmbeddings = []
knownNames = []

#total number of faces processed
total = 0

#loop over image paths
for (i, imagePath) in enumerate (imagePaths):
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    #load image, resize to 600 pixels (maintains aspect ratio)
    #and grab the image dimension
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    #construct blob from image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    #apply opencv's face detector to localise faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    #process the detections
    if len(detections) > 0:
        #assuming that each image has only one face,
        #we find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        #ensure that the detections with the largest prob also meets
        #the minimum prob (filtering out weak detections)
        if confidence > args["confidence"]:
            #compute the xy coordinates of the bounding box of the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #extract the face ROI and grab its dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            #ensure that the face w and h are suffi large
            if fW < 20 or fH < 20:
                continue

            #construct blob for the face ROI, pass the blob
            #through the face embedding model to get a 128-d embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255),
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            #add the name of the person + face embedding to list
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

#dump data to disk
print("[INFO] serialising {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names" : knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dump(data))
f.close()
