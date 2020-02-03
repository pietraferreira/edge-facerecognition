import os
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset', help='File path of dataset.', required=True)
parser.add_argument('--output_dir', help='File path of output folder.', required=True)
args = parser.parse_args()

def create_folders(input_dir, train_output_dir, test_output_dir):
  training_classes = {}
  for class_name in os.listdir(input_dir):
      train_folder_name = os.path.join(train_output_dir, class_name)
      test_folder_name = os.path.join(test_output_dir, class_name)
      if not os.path.exists(train_folder_name):
          os.makedirs(train_folder_name)
      if not os.path.exists(test_folder_name):
          os.makedirs(test_folder_name)
      class_dir_name = os.path.join(input_dir, class_name)
      for image_name in os.listdir(class_dir_name):
          input_file_name = os.path.join(class_dir_name, image_name)
          train_output_file_name = os.path.join(train_folder_name, image_name)
          test_output_file_name = os.path.join(test_folder_name, image_name)
          if class_name in training_classes and training_classes[class_name] >= 19:
              copyfile(input_file_name, test_output_file_name)
          else:
              copyfile(input_file_name, train_output_file_name)
              if class_name not in training_classes:
                  training_classes[class_name] = 1
              else:
                  training_classes[class_name] += 1

input_dir = args.input_dataset
train_output_dir = os.path.join(args.output_dir, "Train")
test_output_dir = os.path.join(args.output_dir, "Test")
if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

create_folders(input_dir, train_output_dir, test_output_dir)




#204
#210
#439
