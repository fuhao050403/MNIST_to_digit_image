import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
from glob import glob
import tensorflow as tf

# Training data
image_directory = "data/train/image/"
label_directory = "data/train/label/"
image_subfolders = glob(image_directory + "*")
#label_filepaths = glob(label_directory + "*.jpg")

with open('data/train/train.csv', 'w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Image", "Label"])

    for image_subfolder in image_subfolders:
        image_filepaths = glob(image_subfolder + "/*.jpg")
        label_filepath = label_directory + image_subfolder[-1] + ".jpg"
        for image_filepath in image_filepaths:
            writer.writerow([image_filepath, label_filepath])

# Testing data
image_directory = "data/test/image/"
label_directory = "data/test/label/"
image_subfolders = glob(image_directory + "*")
#label_filepaths = glob(label_directory + "*.jpg")

with open('data/test/test.csv', 'w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Image", "Label"])

    for image_subfolder in image_subfolders:
        image_filepaths = glob(image_subfolder + "/*.jpg")
        label_filepath = label_directory + image_subfolder[-1] + ".jpg"
        for image_filepath in image_filepaths:
            writer.writerow([image_filepath, label_filepath])