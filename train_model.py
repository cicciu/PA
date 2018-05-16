import os
import sys
import glob

import dlib
from skimage import io

""" 
Execute line example:
python train_model.py xml_files/dataset_rect_vertical.xml models/rect_vertical_dlt.svm
"""

path_xml_file = sys.argv[1]
path_model = sys.argv[2]

# create options for simple_detector
options = dlib.simple_object_detector_training_options()
# Since rect are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True

# train model
dlib.train_simple_object_detector(path_xml_file, path_model, options)


# It will print(the precision, recall, and then) average precision.
print("") 
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(path_xml_file, path_model)))
