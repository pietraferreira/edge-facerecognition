import os
import tensorflow as tf
from PIL import Image
import numpy as np

dimen = 128

dir_path = input("Enter images directory path: ")
out_path = input("Enter images output path: ")

sub_dir_list = os.listdir(dir_path)
images = list()
labels = list()

for i in range(len(sub_dir_list)):
    label = i
    image_names = os.listdir(os.path.join(dir_path, sub_dir_list[i]))
