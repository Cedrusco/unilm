import os, uuid, base64, torch, glob, time, logging, subprocess
from examples.classification.predict import convert_hocr_to_feature
from layoutlm.data.convert import convert_img_to_xml
from layoutlm.modeling.layoutlm import LayoutlmConfig, LayoutlmForSequenceClassification
from layoutlm.data.rvl_cdip import CdipProcessor
from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
from examples.classification.mapping import get_label, check_if_exists, add_template_id

logger = logging.getLogger(__name__)
MODEL_DIR = 'aetna-trained-model'
BASE_MODEL_DIR = 'models/layoutlm-base-uncased'
TIFF_DIR = 'data/tiffs'
XML_DIR = 'data/images'
LABEL_DIR = 'data/labels'


# Decodes & converts image to .xml file, adding file and template ID to train.txt
# Expects two arguments, template_id - template ID that corresponds to image being added to image.txt,
# base64_img - base64 image string to convert to .xml and add to dataset
# Returns filename of outputted .xml file
def add_data(template_id, base64_img):
    xml_path = os.path.join(XML_DIR, template_id)
    img_path = os.path.join(TIFF_DIR, template_id)
    file_name = uuid.uuid4().hex
    try:
        os.mkdir(img_path)
    except:
        pass
    try:
        os.mkdir(xml_path)
    except:
        pass
    img = os.path.join(TIFF_DIR, template_id, f'{file_name}.tiff')
    with open(img, 'wb') as file_to_save:
        decoded_image_data = base64.b64decode(base64_img, '-_')
        file_to_save.write(decoded_image_data)
    convert_img_to_xml(img, xml_path, file_name)
    xml_file = os.path.join(xml_path, f'{file_name}.xml')
    print(f'xml file {xml_file}')
    add_training_label(f'{template_id}/{file_name}.xml', template_id)
    return xml_file


# Adds a template ID/file entry to train.txt
# Expects two arguments, file_path - path of .xml file to add,
# template_id - template ID of file to add
def add_training_label(file_path, template_id):
    training_labels_file = os.path.join(LABEL_DIR,'train.txt')
    label = get_label(template_id)
    with open(training_labels_file, 'a+') as file_object:
        file_object.write('\n')
        file_object.write(f'{file_path} {label}')


# Update the version of model in version.txt, if template ID exists already, increment 1.0 -> 1.1
# If template ID does not exist, increment 1.1 -> 2.0
# Expects one argument, id_exists - boolean designating whether or not template ID exists, returned by check_if_exists()
# Returns updated version number
def update_version(id_exists):
    f = open("data/labels/version.txt", "r")
    text = f.read()
    version = text.split()
    model_version, sub_model_version = version[1].split('.')
    if id_exists:
        new_sub_model_version = int(sub_model_version) + 1
        new_version = f'{model_version}.{new_sub_model_version}'
        text = f'version {new_version}'
        f = open("data/labels/version.txt", "w")
        f.write(text)
        return new_version
    else:
        new_model_version = int(model_version) + 1
        new_version = f'{new_model_version}.0'
        text = f'version {new_version}'
        f = open("data/labels/version.txt", "w")
        f.write(text)
        return new_version


# Trains the model against a new image and template ID - if template ID doesn't exist, we call do_retrain()
# Otherwise, we call cont_train()
# Expects two arguments, base64_img - base64 image to train model against,
# template_id - template ID of document
def do_training(base64_img, template_id):
    subprocess.Popen("cd ../../; python setup.py install", shell=True).wait()
    template_exists = check_if_exists(template_id)
    update_version(template_exists)
    if template_exists:
        print('do_training exists ', template_id)
        label = get_label(template_id)
        return cont_train(base64_img, template_id, label)
    else:
        label = add_template_id(template_id)
        print('do_training does not exists ', template_id)
        do_retrain(base64_img, template_id, label)


# Leverages pre-trained model and trains against a new image with recognized template_id,
# Expects three arguments, base64_img - base64 image to train model against
# template_id - template ID of document,
# label - label of document
# Returns the directory of the newly trained model
def cont_train(base64_img, template_id, label):
    config = LayoutlmConfig.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = LayoutlmForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
    processor = CdipProcessor()
    label_list = processor.get_labels()
    hocr_file = add_data(template_id, base64_img)
    feature = convert_hocr_to_feature(hocr_file, tokenizer, label_list, label)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=5e-5, eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=40
    )
    epoch_count = 40
    model.zero_grad()

    for _ in range(epoch_count):
        model.train()
        inputs = {
            "input_ids": torch.tensor([feature.input_ids]),
            "attention_mask": torch.tensor([feature.attention_mask]),
            "token_type_ids": torch.tensor([feature.token_type_ids]),
            "labels": torch.tensor([feature.label]),
            "bbox": torch.tensor([feature.bboxes])
        }
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print('save model')
    save_model(model, tokenizer, MODEL_DIR)
    return {"trained_model_name": MODEL_DIR}


# Saves a pre-trained model to designated directory, leverages save_pretrained() from modeling_utils.py
# and save_pretrained() from tokenization_utils.py
# Expects three arguments, model - pre-trained model to save,
# tokenizer - pre-trained tokenizer to save,
# output_dir - directory to save in
def save_model(model, tokenizer, output_dir):
    output_dir = os.path.join(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('output_dir %s ', output_dir)
    logger.info("Saving model checkpoint to %s", output_dir)
    model_class = LayoutlmForSequenceClassification
    tokenizer_class = BertTokenizerFast
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(output_dir, os.path.join(output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(output_dir)
    tokenizer = tokenizer_class.from_pretrained(
        output_dir, do_lower_case=True
    )
    model.to('cpu')


# Removes cached model file from project directory
def remove_cache():
    for filename in glob.glob("data/cached*"):
        os.remove(filename)


# Leverages base layoutLM model and runs training on entire dataset, including the newly added image
# Expects three arguments, base64_img - base64 image to train model against
# template_id - template ID of document
# label - label of document
def do_retrain(base64_img, template_id, label):
    remove_cache()
    time.sleep(10) 
    add_data(template_id, base64_img)
    time.sleep(10)
    subprocess.Popen("python run_classification.py  --data_dir data \
                              --model_type layoutlm \
                              --model_name_or_path models/layoutlm-base-uncased \
                              --output_dir aetna-trained-model \
                              --do_lower_case \
                              --max_seq_length 512 \
			                  --do_train \
                              --num_train_epochs 40.0 \
                              --logging_steps 5000 \
                              --save_steps 5000 \
                              --per_gpu_train_batch_size 1 \
                              --per_gpu_eval_batch_size 1 \
			                  --overwrite_output_dir", shell=True)       
                              

if __name__ == "__main__":
    do_retrain("image", "label", "label")
    # update_version(True)

