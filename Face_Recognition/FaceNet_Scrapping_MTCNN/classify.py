from numpy import load
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from random import choice
from numpy import expand_dims
from matplotlib import pyplot

data = load('compressed_face-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("Dataset: train = %d, test = %d" % (trainX.shape[0], testX.shape[0]))

#normalise the face embedding vectors
#meaning scalling values until the lenght of the vectors is 1 or unit length
in_encoder = Normalizer()
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

#convert the string target variables for each name to integer
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

#fit the model using SVM as it seems to be very effective at separating
#the face embedding vectors
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

#predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

#score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

#summarise
print('Accuracy = train=%.3f, test=%.3f' % (score_train*100, score_test*100))

'''PLOTTING'''
#load the faces in the test dataset
data_ = load('compressed_data.npz')
testX_faces = data_['arr_2']

#select random example from test set, get embeddings, face pixels
#expected class prediction and the corresponding name of the class
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

#use the face embedding as an input to make a single prediction
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

#get the name for the predicted class integer and prob for this prediction
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

#printing everything
print("Predicted: %s (%.3f)" % (predict_names[0], class_probability))
print("Expected: %s" % random_face_name[0])

#acc plotting now
pyplot.imshow(random_face_pixels)
title = "%s (%.3f)" % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
