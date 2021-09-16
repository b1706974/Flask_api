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
prototxt_path = os.path.join('deploy.prototxt')
caffemodel_path = os.path.join('res10_300x300_ssd_iter_140000.caffemodel')
# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
url = 'http://a294-14-240-28-81.ngrok.io/data'
client = pymongo.MongoClient(
    "mongodb+srv://Buudao123:Buudao9699@cluster0.1y49z.mongodb.net/API?retryWrites=true&w=majority")
db = client.API
image = db.image


def chuyen_base64_sang_anh(anh_base64):
    try:
        anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return anh_base64


def detect_face(face):
    count = 0
    for file in os.listdir('images'):
        file_name, file_extension = os.path.splitext(file)
        if file_extension in ['.png', '.jpg']:
            image = cv2.imread('images/' + file)
            (h, w) = face.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(face, (400, 400)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            model.setInput(blob)
            detections = model.forward()
            # Identify each face
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                confidence = detections[0, 0, i, 2]
                # If confidence > 0.5, save it as a separate file
                if confidence > 0.5:
                    count += 1
                    frame = face[startY:endY, startX:endX]
                    cv2.imwrite('faces/' + str(i) + '_' + file, frame)
    # zip tat ca anh trong file lai
    shutil.make_archive("data", 'zip', "./faces")
    #####Xoa het anh trong may#####
    mypath = "C:/Users/MyPC/Desktop/flask_api/faces"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))


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
    # zip tat ca anh trong file lai
    shutil.make_archive("data", 'zip', "./augmented")
    #####Xoa het anh trong may#####
    mypath = "C:/Users/MyPC/Desktop/flask_api/augmented"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))


###########################################################################
@app.route('/face', methods=['GET', 'POST'])
@cross_origin(origin='*')
def get_process():
    facebase64 = request.form.get('facebase64')
    face = chuyen_base64_sang_anh(facebase64)
    detect_face(face)
    for name in glob.glob("C:/Users/MyPC/Desktop/flask_api/*.zip"):  # base64Zip
        with open(name, "rb") as image_file:
            resultt = base64.b64encode(image_file.read()).decode()
    response = jsonify({
        'message': str(resultt)
    })
    return response

###########################################################################
@app.route('/data', methods=['GET', 'POST'])
@cross_origin(origin='*')
def data_aug():
    facebase64 = request.form.get('facebase64')
    imgdata = base64.b64decode(facebase64)
    filename = './templates/1.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    db.image.replace_one(
        {"Name": 'done'},
        {
            "Name": 'done',
            "facebase64": str(facebase64),
        }
    )
    aug_data()
    new_data = {}
    for name in glob.glob("C:/Users/MyPC/Desktop/flask_api/*.zip"):  # base64Zip
        with open(name, "rb") as image_file:
            result = base64.b64encode(image_file.read()).decode()
    response = jsonify({
        'message': str(result),
    }
    )
    print(response)

###########################################################################
@app.route('/', methods=['GET', 'POST'])
@cross_origin(origin='*')
def main():
    return render_template('hello.html')


###########################################################################
# startbackend
if __name__ == '__main__':
    app.run(host='127.0.0.1', port='6868', debug=True)
