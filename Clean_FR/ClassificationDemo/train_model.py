from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recogniser", required=True,
	help="path to output model trained to recognise faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

#load embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

#encode labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["name"])

#train the model to take 128-d embeddings and
#produce the face recognition
print("[INFO] training model...")
recogniser = SVC(C=1.0, kernel="linear", probability=True)
recogniser.fit(data["embeddings"], labels)

#output the model and labels encoder to disk
f = open(args["recogniser"], "wb")
f.write(pickle.dumps(recogniser))
f.close()

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
