import os
from shutil import copyfile
import random

index_age = 0
index_sex = 1
index_race = 2

index = index_race

def create_folders(input_dir, train_output_dir, test_output_dir):
  classes = []

  for file in os.listdir(input_dir):
      class_name = file.split("_")[index]
      if class_name not in classes:
          classes.append(class_name)
  for class_name in classes:
      train_folder_name = os.path.join(train_output_dir, class_name)
      test_folder_name = os.path.join(test_output_dir, class_name)
      if not os.path.exists(train_folder_name):
          os.makedirs(train_folder_name)
      if not os.path.exists(test_folder_name):
          os.makedirs(test_folder_name)
  training_classes = []
  for file in os.listdir(input_dir):
      class_name = file.split("_")[index]
      input_file_name = os.path.join(input_dir, file)
      train_output_file_name = os.path.join(train_output_dir, class_name, file)
      test_output_file_name = os.path.join(test_output_dir, class_name, file)
      if class_name in training_classes:
          copyfile(input_file_name, test_output_file_name)
      else:
          copyfile(input_file_name, train_output_file_name)
          training_classes.append(class_name)

input_dir = "/Users/pietraferreira/Downloads/UTKFace"
train_output_dir = "/Users/pietraferreira/Downloads/Train"
test_output_dir = "/Users/pietraferreira/Downloads/Test"
create_folders(input_dir, train_output_dir, test_output_dir)

#and (training_classes[class_name] >= 200 or random.random() < 0.9):

#if class_name not in training_classes:
    #training_classes[class_name] = 1
#else:
    #training_classes[class_name] += 1
