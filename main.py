import glob
import os
import pymongo
import requests
from flask import Flask, jsonify, flash, url_for, render_template, send_from_directory
from flask import request
from flask_cors import CORS, cross_origin
import base64
import cv2
from PIL import Image
from base64 import decodestring
from pywin.framework.mdi_pychecker import dirpath
from skimage import io
from keras_preprocessing.image import ImageDataGenerator
from werkzeug.utils import redirect, secure_filename
import shutil
import requests
from pprint import pprint

app = Flask(__name__)


def decor_base64(anh_base64):
    try:
        anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return anh_base64


def aug_data():
    datagen = ImageDataGenerator(
        rotation_range=45,  # 0-45
        width_shift_range=0.2,  # % shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')  # nearest, constant, reflect, wrap

    x = io.imread('./templates/1.jpg')  # shape (256, 256, 3)

    x = x.reshape((1,) + x.shape)  # shape (1, 256, 256, 3)

    i = 0
    for batch in datagen.flow(x, batch_size=16,
                              save_to_dir='augmented',
                              save_prefix='aug',
                              save_format='jpg'):
        i += 1
        if i > 20:
            break
    shutil.make_archive("data", 'zip', "./augmented")
    #####autodeletefunc#####
    mypath = "C:/Users/MyPC/Desktop/flas/augmented"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))


###########################################################################
@app.route('/data', methods=['GET', 'POST'])
@cross_origin(origin='*')
def get_process():
    facebase64 = request.form.get('facebase64')
    imgdata = base64.b64decode(facebase64)
    filename = './templates/1.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    aug_data()
    new_data = {}
    for name in glob.glob("C:/Users/MyPC/Desktop/flas/*.zip"):  # base64Zip
        with open(name, "rb") as image_file:
            result = base64.b64encode(image_file.read()).decode()
    print(request.json)
    return {'status': 200,
            'message': str(result),
            }


###########################################################################
# startbackend
if __name__ == '__main__':
    app.run(debug=True)
