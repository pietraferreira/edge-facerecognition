# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo to classify image."""

import argparse
import re
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import numpy as np

# Function to read labels from text files.
def ReadLabelFile(file_path):
  """Reads labels from text file and store it in a dict.

  Each line in the file contains id and description separted by colon or space.
  Example: '0:cat' or '0 cat'.

  Args:
    file_path: String, path to the label file.

  Returns:
    Dict of (int, string) which maps label id to description.
  """
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = re.split(r'[:\s]+', line.strip(), maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

#CUSTOM
def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

metric = "cosine" #euclidean or cosine

threshold = 0
if metric == "euclidean":
    threshold = 0.35
elif metric == "cosine":
    threshold = 0.07

def verifyFace(emb1, emb2):
    #produce 128-dimensional representation
    if metric == "euclidean":
        #emb1 = l2_normalize(emb1)
       # emb2 = l2_normalize(emb2)
       
        #dist = np.linalg.norm(emb1 - emb2)
        euclidean_distance = findEuclideanDistance(emb1, emb2)
        print("euclidean distance (l2 norm): ", euclidean_distance)

        if euclidean_distance < threshold:
            print("verified... they are same person")
        else:
            print("unverified! they are not same person!")

    elif metric == "cosine":
        cosine_similarity = findCosineSimilarity(emb1, emb2)
        print("cosine similarity: ",cosine_similarity)

        if cosine_similarity < 0.07:
            print("verified... they are same person")
        else:
            print("unverified! they are not same person!")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument(
      '--label', help='File path of label file.', required=True)
  parser.add_argument(
      '--image', help='File path of the image to be recognized.', required=True)
  parser.add_argument(
      '--image2', help='File path to the second image to be compared', required=True)
  args = parser.parse_args()
        

  # Prepare labels.
  labels = ReadLabelFile(args.label)
  # Initialize engine.
  engine = ClassificationEngine(args.model)
  engine2 = ClassificationEngine(args.model)
  # Run inference.
  img = Image.open(args.image)
  result = engine.ClassifyWithImage(img, top_k=1)
  img2 = Image.open(args.image2)
  result2 = engine2.ClassifyWithImage(img2, top_k=1)
  #print(result)
  #print("RESULT 2:")
  #print(result2)
  verifyFace(result, result2)
  #for result in engine.ClassifyWithImage(img, top_k=1):
   # print('---------------------------')
    #print(result)
    #print('Score : ', result[1])

if __name__ == '__main__':
  main()
