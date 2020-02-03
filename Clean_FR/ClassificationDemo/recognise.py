import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recogniser", required=True,
	help="path to model trained to recognise faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load face detector from disk
print(["[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#load face embedding model from disk
print("[INFO] loading face recogniser...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

#load the actual face recognition model along with the le
recogniser = pickle.loads(open(args["recogniser"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

#load image, resise it and grab dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

#construct blob from image
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False
)

#apply opencvs face detector
detector.setInput(imageBlob)
detections = detector.forward()

#loop over the detections
for i in range(0, detections.shape[2]):
    #extract the probability associated w the prediction
    confidence = detections[0, 0, i, 2]

    #filter out weak detections
    if confidence > args["confidence"]:
        #compute the coordinates for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #extract the face ROI
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        #ensure the width and height are suffi large
        if fW < 20 or fH < 20:
            continue

        #construct blob for the ROI, pass the blob through
        #the embedding model to obtain the 128-d embedding
        faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        #perform classification to recognise faces
        preds = recogniser.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        #draw the bounding box with the probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

#show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)