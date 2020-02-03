from numpy import load
from numpy import expand_dims
from keras.models import load_model
from numpy import asarray
from numpy import savez_compressed

data = load('compressed_data.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("Loaded: \n",
      'trainX: ', trainX.shape, '\n',
      'trainy: ', trainy.shape, '\n',
      'testX: ', testX.shape, '\n',
      'testy: ', testy.shape, '\n',
      )

#load the model
model = load_model('model/facenet_keras.h5')
print("Model loaded successfully!")

#get the face embeddings for one face
def get_embedding(model, face_pixels):
    #scale the pixel values
    face_pixels = face_pixels.astype('float32')
    #standardize pixel values across channels
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    #transform face into one single sample
    samples = expand_dims(face_pixels, axis=0)
    #make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

#convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)

newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)

#save arrays to a compressed file
savez_compressed('compressed_face-embeddings.npz', newTrainX, trainy, newTestX, testy)
