from PIL import Image
from flask import Flask, request,flash, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from src.config.upload import *
import os
import cv2
from src.utils.utils import generateUniquePrefix
from src.utils.torch_utils import get_prediction,img_transform

app = Flask(__name__)


@app.route("/")
def HelloWorld():
    return "Hi there!"


@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({
            'code':404,
            'data':'No file found!'
        })
    file = request.files['file']
    # If user does not select a file, the browser submits an
    # empty file without a filename
    if file.filename == '':
        return jsonify({
            'code': 404,
            'data': "No selected file!"
        })
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uniquePrefix = generateUniquePrefix()
        os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'],"images",uniquePrefix))
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'],"images",uniquePrefix,filename)
        file.save(saved_path)
        
        # Take already saved images to making prediction
        img = cv2.imread(saved_path)
        transformImg = img_transform(img)
        prediction = get_prediction(transformImg)
        data = {'prediction': prediction.item(), 'class_name':str(prediction.item())}
        return jsonify({
            'code':200,
            'data':data
        })
        
    
@app.errorhandler(404)
def route_not_found(e):
    return jsonify({
        'code':404,
        'data':'Route not found on server!'
    })

if __name__ == '__main__':
    createUploadFolders()
    
    # config folder
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    app.run(debug=True,host="localhost", port=8888)