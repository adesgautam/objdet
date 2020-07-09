
import os
import sys

import argparse
from yolo_files.Utils.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from yolo_files.Utils.utils import load_extractor_model, load_features, parse_input, detect_object
import pandas as pd
import numpy as np
from yolo_files.Utils.get_file_paths import GetFileList
import random

print("Starting...")

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

input_path = "test_images"
output_path = "test_images_results"

anchors_path = "yolo_files/yolo_anchors.txt"
classes_path = "yolo_files/data_classes.txt"
model_path = "yolo_files/model/checkpoint_5.h5"
detection_results_file = output_path+"/Detection_Results.csv"

score = 0.25
gpu_num = 1
no_save_img = False
postfix = "_text"
file_types = [".jpg", ".jpeg", ".png", ".mp4"]
save_img = not no_save_img

print("Checking files...")
# Check file types
if file_types:
    input_paths = GetFileList(input_path, endings = file_types)
else:
    input_paths = GetFileList(input_path)

print("File:", input_paths)
# Split images and videos
img_endings = ('.jpg','.jpg','.png')
vid_endings = ('.mp4','.mpeg','.mpg','.avi')

input_image_paths = [] 
input_video_paths =[]
for item in input_paths:
    if item.endswith(img_endings):
        input_image_paths.append(item)
    elif item.endswith(vid_endings):
        input_video_paths.append(item)    

if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Loading Model...")
# Define YOLO detector
yolo = YOLO(**{"model_path": model_path,
                "anchors_path": anchors_path,
                "classes_path": classes_path,
                "score" : score,
                "gpu_num" : gpu_num,
                "model_image_size" : (416, 416),
                })
print("Model Loaded!")
# Make a dataframe for the prediction outputs
out_df = pd.DataFrame(columns=['image', 'image_path','xmin', 'ymin', 'xmax', 'ymax', 'label','confidence','x_size','y_size'])

# Labels to draw on images
class_file = open(classes_path, 'r')
input_labels = [line.rstrip('\n') for line in class_file.readlines()]
print('Found {} input labels: {} ...'.format(len(input_labels), input_labels))

if input_image_paths:
    print('Found {} input images: {} ...'.format(len(input_image_paths), [ os.path.basename(f) for f in input_image_paths[:5]]))
    start = timer()
    text_out = ''

    # This is for images
    for i, img_path in enumerate(input_image_paths):
        print(img_path)
        prediction, image = detect_object(yolo, img_path, save_img = save_img,
                                          save_img_path = output_path,
                                          postfix=postfix)
        y_size, x_size, _ = np.array(image).shape
        for single_prediction in prediction:
            out_df=out_df.append(pd.DataFrame([[os.path.basename(img_path.rstrip('\n')),img_path.rstrip('\n')]+single_prediction + [x_size,y_size]], columns=['image','image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label','confidence','x_size','y_size']))
    end = timer()
    print('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(
         len(input_image_paths), end-start, len(input_image_paths)/(end-start)
         ))
    out_df.to_csv(detection_results_file,index=False)


# This is for videos
if input_video_paths:
    print('Found {} input videos: {} ...'.format(len(input_video_paths), [ os.path.basename(f) for f in input_video_paths[:5]]))
    start = timer()
    for i, vid_path in enumerate(input_video_paths):
        output_path = os.path.join(FLAGS.output,os.path.basename(vid_path).replace('.', postfix+'.'))
        detect_video(yolo, vid_path, output_path=output_path)
    
    end = timer()
    print('Processed {} videos in {:.1f}sec'.format(len(input_video_paths), end-start))

# Close the current yolo session
yolo.close_session()



