
import os
import cv2
import sys
import random
import math
import re
import numpy as np
import pandas as pd 
import warnings
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
import ntpath
import itertools
import urllib
import json
import base64
import string
import keras 

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from tensorflow.python.keras.backend import set_session

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "mydataset"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + class1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Directory to save logs and trained model
MODEL_DIR = 'logs'
custom_WEIGHTS_PATH = "models/mask_rcnn_trained_weights.h5"  # TODO: update this path

global graph
graph = tf.get_default_graph() 

config = BalloonConfig()
dataset = {}

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

DEVICE = "/cpu:0"
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

session = tf.Session()
init = tf.global_variables_initializer()
set_session(session)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)


def model_predict(image_path):
    image = skimage.io.imread(image_path)

    dataset['class_names'] = ['BG','class1'] 
    results = ''
    with graph.as_default():
        results = model.detect([image], verbose=1)
    ax = get_ax(1)
    r = results[0]
    masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset['class_names'], r['scores'], ax=ax,
                            title="Predictions")
    return masked_image










