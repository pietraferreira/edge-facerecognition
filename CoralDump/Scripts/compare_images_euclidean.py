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

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

threshold = 0.35

def verifyFace(emb1, emb2):
    emb1 = l2_normalize(emb1)
    emb2 = l2_normalize(emb2) 

    euclidean_distance = findEuclideanDistance(emb1, emb2)
    print("Euclidean distance: ", euclidean_distance)
    if euclidean_distance < threshold:
        print("Match!")
    else:
        print("No match!")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument(
      '--image1', help='File path of the first image to be compared.', required=True)
  parser.add_argument(
      '--image2', help='File path to the second image to be compared', required=True)
  args = parser.parse_args()

  # Initialize engine.
  engine1 = EmbeddingEngine(args.model)
  engine2 = EmbeddingEngine(args.model)
  # Run inference.
  img1 = Image.open(args.image1)
  result1 = engine1.CreateEmbeddingsFromImage(img1)
  img2 = Image.open(args.image2)
  result2 = engine2.CreateEmbeddingsFromImage(img2)
  verifyFace(result1, result2)

if __name__ == '__main__':
  main()
