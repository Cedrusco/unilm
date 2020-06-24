#!flask/bin/python
import os
from flask import Flask, request, jsonify, abort
from waitress import serve
from flask_cors import CORS
import base64
import binascii
import sys
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)
from examples.classification.predict_api import predict
from examples.classification.train import do_training
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
from multiprocessing import Process

@app.route('/predict', methods=['POST'])
def predict_label():
    if not request.json or not 'img' in request.json:
        abort(400)
    body_img = request.json.get("img")
    if not body_img:
        abort(400)
    try:
        base64.b64decode(body_img, '-_')
    except:
        return "Not a base-64", 404
    matches = request.json['numMatches'] if 'numMatches' in request.json else 1
    response = predict(body_img, matches)
    return jsonify(response), 201
@app.route('/train', methods=['POST'])
def train_label():
    if 'testing' in request.json and request.json.get("testing") == True:
        return "Reached Endpoint!", 200
    if not request.json or not 'img' in request.json or not 'template_id' in request.json:
        abort(400)  
    body_img = request.json.get("img")
    body_template_id = request.json.get("template_id")
    if not body_img or not body_template_id or body_img.isspace() or body_template_id.isspace():
        abort(400)
    try:
        base64.b64decode(body_img, '-_')
    except:
        return "Not a base-64", 404
    task_training = Process(target=do_training, args=( body_img, body_template_id))
    task_training.start()
    return "Training in process!", 202

if __name__ == '__main__':
    app.run(port=6060, debug=True, host='0.0.0.0')
