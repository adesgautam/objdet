
from flask import Flask, request, render_template, send_from_directory
from shutil import rmtree
from flask_caching import Cache

import os
import cv2

from utils.mask_model import model_predict

__author__ = 'Adesh Gautam'

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# cache = Cache()
cache = Cache()
cache.init_app(app)
with app.app_context():
    cache.clear()

# Setup
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print("APP_ROOT", APP_ROOT)

images = []
images_show = []

# Utils
def save_file(file, target):
    filename = file.filename
    destination = "/".join([target, filename])
    file.save(destination)
    return destination

def get_masked_image(img_path):
    mask_img = model_predict(img_path)
    return mask_img

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    jpg_file = request.files.getlist("jpg")[0]
    if jpg_file.filename=='':
        return render_template("upload.html", msg="Please choose atleast one file.")

    # Check for target directory
    target = os.path.join(APP_ROOT, 'images/')
    if os.path.isdir(target):
        rmtree(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target), "\nDirectory exists!")

    # Save file1 field images
    print(jpg_file)
    destination = save_file(jpg_file, target)
    print("Destination:", destination)

    # get masked image
    # image = cv2.imread(destination)
    masked_image = get_masked_image(destination)
    masked_img_path = 'images/masked_image.jpg'
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(masked_img_path, masked_image)

    data = {destination.split('/')[-1]: masked_img_path.split('/')[-1]}

    return render_template("results.html", image_data=data)

@app.route('/<filename>')
def send_image(filename):
    return send_from_directory("images/", filename)

if __name__ == "__main__":
    app.run(port=4399, debug=False, threaded=True)






    