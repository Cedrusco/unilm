import os, uuid, base64
import sys
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)
from layoutlm.data.convert import convert_img_to_xml
from examples.classification.predict import make_prediction
from examples.classification.mapping import  get_template_id

# assumes model exists and is in same directory as this file
MODEL_DIR = 'aetna-trained-model'
OUTPUT_DIR = 'output'


# Accepts a base64 image and leverages pre-trained model to predict top x labels on the image
# Expects two arguments, base64_img - base64 encoded image to predict on
# num_matches - number of matches to return
def predict(base64_img, num_matches):
    try:
        os.mkdir(OUTPUT_DIR)
    except:
        pass
    filename = uuid.uuid4().hex
    # assumes that base64_img encodes a .tiff file
    img = os.path.join(OUTPUT_DIR, filename + '.tiff')
    with open(img, 'wb') as file_to_save:
        decoded_image_data = base64.b64decode(base64_img, '-_')
        file_to_save.write(decoded_image_data)
    convert_img_to_xml(img, OUTPUT_DIR, filename)
    hocr = os.path.join(OUTPUT_DIR, filename + '.xml')
    matches = make_prediction(MODEL_DIR, hocr, num_matches)
    match_array = []
    f = open("data/labels/version.txt", "r")
    text = f.read()
    version = text.split()

    for rank, label, prob in matches:
        match = {
            'rank': rank,
            'template_id': get_template_id(label),
            'confidence': prob
        }
        match_array.append(match)
    response = {
        'model version': version[1],
        'matches': match_array,
    }
    return response

if __name__ == "__main__":
    predict('', 1)