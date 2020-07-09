import os
import subprocess
import time
import requests
import progressbar

download_folder = "models"
yolo3_weights = download_folder+'/yolov3.weights'

# Set URL
url = 'https://pjreddie.com/media/files/yolov3.weights'
r = requests.get(url,stream=True)

# Download yolo weights
f = open(yolo3_weights, 'wb')
file_size = int(r.headers.get('content-length'))
chunk = 100
num_bars = file_size // chunk
bar =  progressbar.ProgressBar(maxval=num_bars).start()
i = 0
for chunk in r.iter_content(chunk):
    f.write(chunk)
    bar.update(i)
    i+=1
f.close()

# Call for weights conversion
convert_file = os.getcwd() + r'/Utils/keras_yolo3/convert.py'
yolo_config  = os.getcwd() + r'/models/yolov3.cfg'
yolo_weights = os.getcwd() + r'/models/yolov3.weights'
yolo_h5      = os.getcwd() + r'/models/yolo.h5'

# Call subprocess for conversion from yolov3.weights to yolo.h5
call_string = 'python ' +convert_file+ ' ' +yolo_config+ ' ' +yolo_weights+ ' ' + yolo_h5
print("Calling subprocess:", call_string)
subprocess.call(call_string , shell=True, cwd = download_folder)