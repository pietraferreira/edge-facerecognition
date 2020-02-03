from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import argparse
from os import listdir
from os.path import isdir
from numpy import savez_compressed
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--image", required=True, help="path to input image")
args = vars(parser.parse_args())

def extract_face(filename, required_size=(160,160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    #bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    #resise to model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

pixels = extract_face(args["image"])

#load faces into a list for a given dir
def load_faces(dir):
    faces = list()
    #enumerate
    for filename in listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces

#load a dataset that contains one subdir for each class that contains imgs
def load_dataset(dir):
    X, y = list(), list()
    #enumerate folders, one per class
    for subdir in listdir(dir):
        path = dir + subdir + '/'
        #skip any files that might be in the dir
        if not isdir(path):
            continue
        faces = load_faces(path)
        #create labels
        labels = [subdir for _ in range(len(faces))]
        #summ progress
        print('Loaded %d examples for class: %s' % (len(faces), subdir))
        #save
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

#load train dataset
trainX, trainy = load_dataset('data/train/')
#load test dataset
testX, testy = load_dataset('data/val/')
#finally save
savez_compressed('compressed_data.npz', trainX, trainy, testX, testy)
