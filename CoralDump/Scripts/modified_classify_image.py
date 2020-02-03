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
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from PIL import Image
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument('--label', help='File path of label file.', required=True)
  parser.add_argument(
      '--dataset', help='File path of the dataset to be recognized.', required=True)
  args = parser.parse_args()

  num_certain = 0
  num_uncertain = 0
  guess_right = 0
  guess_wrong = 0
 
  engine = ClassificationEngine(args.model)
  labels = dataset_utils.ReadLabelFile(args.label)

  for folder_name in os.listdir(args.dataset):
      folder_path = os.path.join(args.dataset, folder_name)
      for image_name in os.listdir(folder_path):
          image_path = os.path.join(folder_path, image_name)
          image = Image.open(image_path)
          result = engine.ClassifyWithImage(image, threshold=0.11, top_k=1)
          if not len(result):
            num_uncertain += 1
            break
          num_certain += 1
          label = labels[result[0][0]]
          print("------------------------")
          print("Expected: ", folder_name)
          print("Actual: ", label, " (certainty ", result[0][1], ")")
          if label != folder_name:
              guess_wrong += 1
          else:
              guess_right += 1
  print("========================")
  print()
  print("Evaluation complete")
  print("Combined accuracy: ", guess_right / (guess_right + guess_wrong))
  print("Total certainty: ", num_certain / (num_certain + num_uncertain))
  print()

if __name__ == '__main__':
  main()
