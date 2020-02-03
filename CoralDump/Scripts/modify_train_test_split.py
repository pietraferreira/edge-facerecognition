import os
import argparse

from shutil import copyfile

parser = argparse.ArgumentParser(description='Resize the train/test split of an existing dataset')
parser.add_argument('--train_data_dir', help='Directory containing the training data')
parser.add_argument('--test_data_dir', help='Directory containing the test data')
parser.add_argument('--train_out_dir', help='Directory to output new training data to')
parser.add_argument('--test_out_dir', help='Directory to output new test data to')
parser.add_argument('--new_size', type=int, help='The desired number of training samples')

args = parser.parse_args()

for class_name in os.listdir(args.train_data_dir):
  num_selected = 0
  
  train_out_path = os.path.join(args.train_out_dir, class_name)
  if not os.path.exists(train_out_path):
    os.makedirs(train_out_path)
  test_out_path = os.path.join(args.test_out_dir, class_name)
  if not os.path.exists(test_out_path):
    os.makedirs(test_out_path)

  # Move samples from training data into training, then testing if/when we have enough samples.
  train_data_path = os.path.join(args.train_data_dir, class_name)
  for sample_name in os.listdir(train_data_path):
    sample_path = os.path.join(train_data_path, sample_name)
    if num_selected < args.new_size:
      output_sample_path = os.path.join(train_out_path, sample_name)
      copyfile(sample_path, output_sample_path)
      num_selected += 1
    else:
      output_sample_path = os.path.join(test_out_path, sample_name)
      copyfile(sample_path, output_sample_path)
  
  # Have we still not selected enough samples? In that case we need to move samples from testing to training too.
  if num_selected < args.new_size:
    test_data_path = os.path.join(args.test_data_dir, class_name)
    for sample_name in os.listdir(test_data_path):
      sample_path = os.path.join(test_data_path, sample_name)
      if num_selected < args.new_size:
        output_sample_path = os.path.join(train_out_path, sample_name)
        copyfile(sample_path, output_sample_path)
        num_selected += 1
      else:
        break

