import os
import argparse

from shutil import rmtree, copytree

parser = argparse.ArgumentParser(description='Reduce the size of a train/test dataset')
parser.add_argument('--train_data_dir', help='Directory containing the training data')
parser.add_argument('--test_data_dir', help='Directory containing the test data')
parser.add_argument('--train_out_dir', help='Directory to output new training dataset to')
parser.add_argument('--test_out_dir', help='Directory to output new test dataset to')
parser.add_argument('--new_size', type=int, help='The desired number of classes')

args = parser.parse_args()

selected_classes = []
selected = 0
for class_name in os.listdir(args.train_data_dir):
  if len(selected_classes) > args.new_size:
    break
  print("Keeping class: ", class_name)
  selected_classes.append(class_name)
  
  train_out_path = os.path.join(args.train_out_dir, class_name)
  # Apparently copytree doesn't like the directory existing already...
  if os.path.exists(train_out_path):
    rmtree(train_out_path)

  test_out_path = os.path.join(args.test_out_dir, class_name)
  # Apparently copytree doesn't like the directory existing already...
  if os.path.exists(test_out_path):
    rmtree(test_out_path)

  train_data_path = os.path.join(args.train_data_dir, class_name)
  copytree(train_data_path, train_out_path)
  test_data_path = os.path.join(args.test_data_dir, class_name)
  copytree(test_data_path, test_out_path)

