from random import choice
from numpy import expand_dims
from matplotlib import pyplot

#load the faces in the test dataset
data = load('compressed_data.npz')
testX_faces = data['arr_2']

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
