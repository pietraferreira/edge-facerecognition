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

"""A demo to compare images."""

import argparse
from edgetpu.embeddings.engine import EmbeddingEngine
from PIL import Image
import numpy as np
import time
import os

# def timer(function):
#     def function_timer(*args, **kwargs):
#         start = time.time()
#         value = function(*args, **kwargs)
#         end = time.time()
#         runtime = end - start
#         msg = "The runtime for {function} took {time} second to complete."
#         print(msg.format(function=function.__name__, time=runtime))
#
#         return value
#     return function_timer




# def get_image_paths(data_dir):
#     classes = None
#     image_paths = []
#     class_idx = 0
#     for root, dirs, files in os.listdir(data_dir):
#       if root == data_dir:
#         # Each sub-directory in `data_dir`
#         classes = dirs
#       else:
#         # Read each sub-directory
#         assert classes[class_idx] in root
#         print('Reading dir: %s, which has %d images' % (root, len(files)))
#         for img_name in files:
#           image_paths.append(os.path.join(root, img_name))
#     return (image_paths, classes)

start = time.time()

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

threshold = 0.07



def verifyFace(emb1, emb2):
    cosine_similarity = findCosineSimilarity(emb1, emb2)
    print("Cosine similarity: ", cosine_similarity)
    if cosine_similarity < threshold:
        print("Match!")
        return True
    else:
        print("No match!")
        return False

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument(
      '--data_dir', help='Path to the dataset.', required=True)
  parser.add_argument(
      '--image2', help='File path to the second image to be compared', required=True)
  args = parser.parse_args()


  root = args.data_dir
  data = []
  labels = []
  data_with_labels = []

  # for (root,dirs,files) in os.walk(root, topdown=True):
  #   for dirs in files:
  #
  #   print(files)
  #
  # for dataset in os.listdir(root):
  #   dataset = list.append()
    # for folder in os.listdir(dataset):
    # #name = list.append(folder)
    #     print(folder)

  for folder in os.listdir(root):
    labels = np.append(labels, folder)

    for file in os.listdir(os.path.join(root, folder)):
        #print(data)
        data = np.append(data, os.path.join(root, folder, file))
        data_with_labels = np.append(data[0] + ",", labels[0])
        # print(data)
  # file = open("testing.txt", "w")
  # file.write(data_with_labels)

  np.savetxt('testing.txt', data_with_labels, fmt='%s')

  # Initialize engine.
  engine1 = EmbeddingEngine(args.model)
  engine2 = EmbeddingEngine(args.model)

  img2 = Image.open(args.image2)
  result2 = engine2.CreateEmbeddingsFromImage(img2)

  num_good = 0
  num_fail = 0

  # Run inference.
  for img in data:
      name = img.split("/")[1]
      name = name.split("_")[0:-1]
      label = "_".join(name)
      print(label)
      if label == "Abel_Pacheco":
          expect_result = True
      else:
          expect_result = False
      opened_img = Image.open(img)

      result = engine1.CreateEmbeddingsFromImage(opened_img)

      np.savetxt(img+".embedding", result)


      if verifyFace(result, result2) == expect_result:
        num_good += 1
      else:
          num_fail += 1


      # verifyFace(result, result2)

  print(num_fail)
  print(num_good)

if __name__ == '__main__':
  main()

end = time.time()
total = (end - start)

#print("Runtime was of {}!".format(total))
