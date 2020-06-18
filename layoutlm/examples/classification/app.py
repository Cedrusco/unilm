#!flask/bin/python
import os
from flask import Flask, request, jsonify, abort
from waitress import serve
from flask_cors import CORS
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
    matches = request.json['numMatches'] if 'numMatches' in request.json else 1
    response = predict(request.json['img'], matches)
    return jsonify(response), 201
@app.route('/train', methods=['POST'])
def train_label():
    if not request.json or not 'img' in request.json or not 'template_id' in request.json:
        abort(400)  
    body_img = request.json.get("img")
    body_template_id = request.json.get("template_id")
    if not body_img or not body_template_id or body_img.isspace() or body_template_id.isspace():
        abort(400)
    task_training = Process(target=do_training, args=( body_img, body_template_id))
    task_training.start()
    return "Training in process!", 202

if __name__ == '__main__':
    # # debug mode
    app.run(host='0.0.0.0', debug=True, port=6060)
    # production mode
    # serve(app, port=6060)