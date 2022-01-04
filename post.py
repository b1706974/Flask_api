import os
import shutil
import zipfile
import requests
from flask import jsonify
import base64
img = open('testm.jpg', 'rb')
data = base64.b64encode(img.read()).decode()
mystr = {
    'img64': data,
        }
url = 'http://127.0.0.1:5000/predict'
r = requests.post(url, data=mystr)
res = r.json()
#print(res['message'])
print(res)
