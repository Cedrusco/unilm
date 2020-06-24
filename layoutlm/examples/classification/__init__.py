# flake8: noqa
from .predict import make_prediction, convert_hocr_to_feature
from .predict_api import predict
from .mapping import get_label, check_if_exists, max_label, add_template_id