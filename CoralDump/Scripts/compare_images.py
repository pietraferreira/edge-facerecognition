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
