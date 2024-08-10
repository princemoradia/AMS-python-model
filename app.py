from flask import Flask, request, jsonify
import cv2 as cv
import os
import numpy as np
from utils.Inceptionresnet import *
import dlib
from mtcnn import MTCNN


app = Flask(__name__)


facenet_encoder = InceptionResNetV2()
facenet_encoder.load_weights("utils/facenet_keras_weights.h5")

FRmodel = facenet_encoder
filepath = "extract_face/"
database = {}

def img_to_encoding(image, model):
    if image is None:
        return 0
    else:
        img = cv.resize(image,(160,160))
        img = np.around(np.asarray(img)/255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = model.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2)

def load_image(filepath):
    for folder in os.listdir(filepath):
        encodings = []
        for subfolder in os.listdir(filepath + folder):
            image_BGR = cv.imread(filepath + folder + "/" + subfolder)
            temp = img_to_encoding(image_BGR, FRmodel)
            encodings.append(temp)
        database[folder] = encodings
    return database

database = load_image(filepath)


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv.GaussianBlur(image, kernel_size, 0)

def apply_histogram_equalization(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    equalized_image = cv.equalizeHist(gray_image)

    equalized_image = cv.normalize(equalized_image, None, 0, 255, cv.NORM_MINMAX)

    return equalized_image
def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)
    min_dist = 100

    for (name, db_enc) in database.items():
        for i in db_enc:
            dist = np.linalg.norm(encoding - i)

            if dist < min_dist:
                min_dist = dist
                identity = name

    return min_dist, identity

def detection(image):

    detector = MTCNN()
    image_rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    print(faces)

    for face in faces:
        x,y,w,h = face['box']
        face1 = image[y:y + h, x:x + w]

    return face1

@app.route('/face_recognition',methods=['POST'])
def face_recognition():
    try:
        image_file = request.files['image']

        tmp_path = "C:/ME/Mini Project/Mini Project/temp_image/tmp_image.jpg"
        image_file.save(tmp_path)

        image = cv.imread(tmp_path)
        image1 = apply_gaussian_blur(image)
        face1 = detection(image1)
        mindist, identity = who_is_it(face1, database, FRmodel)

        if mindist < 0.7:
            response = {"result": {"identity": identity, "distance": float(mindist)}, "status": "success"}
        else:
            response = {"result": None, "status": "unknown"}

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



@app.route('/')
def hello_world():
    return "hello world"

if __name__ == '__main__':
    app.run(host='192.168.74.115', port=5000, debug=True)
