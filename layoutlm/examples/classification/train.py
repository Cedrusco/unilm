import os, uuid, base64, torch
import logging
from examples.classification.predict import convert_hocr_to_feature
from layoutlm.data.convert import convert_img_to_xml
from layoutlm.modeling.layoutlm import LayoutlmConfig, LayoutlmForSequenceClassification
from layoutlm.data.rvl_cdip import CdipProcessor, get_prop, DocExample, convert_examples_to_features
from transformers import BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
logger = logging.getLogger(__name__)
MODEL_DIR = 'aetna-trained-model'
OUTPUT_DIR = 'output'


def prepare_image(base64_img):
    # returns path of converted hocr file
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
    convert_img_to_xml(img, OUTPUT_DIR)
    return os.path.join(OUTPUT_DIR, filename + '.xml')


def do_training(base64_img, label):
    config = LayoutlmConfig.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = LayoutlmForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

    processor = CdipProcessor()
    label_list = processor.get_labels()
    hocr_file = prepare_image(base64_img)
    # todo: need to talk about how labels are defined and how we feed them in with the 2 different cases (label exists/not)
    feature = convert_hocr_to_feature(hocr_file, tokenizer, label_list, label)

    # from run_classification.py, some parameters are filled in with default value according to training_args.py
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
        optimizer, num_warmup_steps=0, num_training_steps=20
    )


    epoch_count = 20
    model.zero_grad()
    # todo: discuss assumptions made on hardware, CPU/GPU
    # todo: discuss training method, just new file? add new file to random sampled data? how do we mitigate overfitting to new input
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
        # todo: (for chris) optimize correctly???? need to confirm functionality
        optimizer.step()
        scheduler.step()
    #todo: (for chris) save model
    save_model(model, tokenizer, MODEL_DIR)
    return { "trained_model_name": MODEL_DIR}


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

if __name__ == "__main__":
    do_training("image", "label")