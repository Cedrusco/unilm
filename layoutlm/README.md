# LayoutLM
**Multimodal (text + layout/format + image) pre-training for document understanding**

## Introduction

LayoutLM is a simple but effective pre-training method of text and layout for document image understanding and information extraction tasks, such as form understanding and receipt understanding. LayoutLM archives the SOTA results on multiple datasets. For more details, please refer to our paper: 

[LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, arXiv Preprint, 2019

## Release Notes

**\*\*\*\*\* New May 16th, 2020: Our LayoutLM paper has been accepted to KDD 2020 as a full paper in the research track\*\*\*\*\***

**\*\*\*\*\* New Feb 18th, 2020: Initial release of pre-trained models and fine-tuning code for LayoutLM v1 \*\*\*\*\***

## Pre-trained Model

We pre-train LayoutLM on IIT-CDIP Test Collection 1.0\* dataset with two settings. 

* LayoutLM-Base, Uncased (11M documents, 2 epochs): 12-layer, 768-hidden, 12-heads, 113M parameters || [OneDrive](https://1drv.ms/u/s!ApPZx_TWwibInS3JD3sZlPpQVZ2b?e=bbTfmM) | [Google Drive](https://drive.google.com/open?id=1Htp3vq8y2VRoTAwpHbwKM0lzZ2ByB8xM)
* LayoutLM-Large, Uncased (11M documents, 2 epochs): 24-layer, 1024-hidden, 16-heads, 343M parameters || [OneDrive](https://1drv.ms/u/s!ApPZx_TWwibInSy2nj7YabBsTWNa?e=p4LQo1) | [Google Drive](https://drive.google.com/open?id=1tatUuWVuNUxsP02smZCbB5NspyGo7g2g)

\*As some downstream datasets are the subsets of IIT-CDIP, we have carefully excluded the overlap portion from the pre-training data.

## Cedrus Trainings
https://drive.google.com/open?id=1SExlh5Ycg8PyFtl-XaJdHmRyvAI2_0_P
Refer to this drive to find all the datasets used in the trainings, models trained, and output of each training.

* Naming convention for the output models: `aetna_dataset_output_"type of pretrained model used"_ "number of epoch"_ "dataset used"`, for example `aetna_dataset_output_base_20_d1`.

* After every training it is recommended to insert the result in the `Research` sheet and upload the output model and results in the `Outputs` folder using the naming convention mentioned above.

## Predict Script
In order to leverage training output/pre-trained model to predict, make sure you are in the conda environment and call:
~~~bash
cd layoutlm/examples/classification
python3 model-predict.py "path/to/output/directory" "path/to/XML/input/file"
~~~

## Tesseract Script
A script to convert dataset of images to properly formatted OCR data is available at `layoutlm/layoutlm/data/convert-OCR.py`

Call script as follows:
~~~bash
cd layoutlm/layoutlm/data
python3 convert-OCR.py "Aetna Dataset -1" (name of dataset directory under layoutlm/layoutlm/data)
~~~

The script will run and make consecutive calls to Tesseract, outputting converted .xml files into a new directory `~/out-OCR` within the original dataset directory.

## Local Set-Up

1- Download the following Folder: https://drive.google.com/drive/folders/1JSlK8pUWag27KfQrxwj-Ufugn9FPOkl-
2- unzip all
3- Add all folders under examples/classification
4- Create a new directory "models" and move layoutlm-base-uncased under it

Setup environment as follows:

~~~bash
conda create -n layoutlm python=3.6
conda activate layoutlm
~~~
~~~bash
## Cuda Setup
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
~~~
~~~bash
## None Cuda Setup
conda install pytorch==1.4.0 -c pytorch
~~~
~~~bash
pip install .
## For development mode
# pip install -e ".[dev]"
python3 setup.py install
~~~

### Document Image Classification Task

We also fine-tune LayoutLM on the document image classification task. You can download the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset from [here](https://www.cs.cmu.edu/~aharley/rvl-cdip/). Because this dataset only provides the document image, you should use the OCR tool to get the texts and bounding boxes. For example, you can easily use Tesseract, an open-source OCR engine, to generate corresponding OCR data in hOCR format. For more details, please refer to the [Tesseract wiki](https://github.com/tesseract-ocr/tesseract/wiki). Your processed data should look like [this sample data](https://1drv.ms/u/s!ApPZx_TWwibInTlBa5q3tQ7QUdH_?e=UZLVFw). 

With the processed OCR data, you can run LayoutLM as follows:

~~~bash
python run_classification.py  --data_dir  data \
                              --model_type layoutlm \
                              --model_name_or_path path/to/pretrained/model/directory \
                              --output_dir path/to/output/directory \
                              --do_lower_case \
                              --max_seq_length 512 \
                              --do_train \
                              --do_eval \
                              --num_train_epochs 40.0 \
                              --logging_steps 5000 \
                              --save_steps 5000 \
                              --per_gpu_train_batch_size 16 \
                              --per_gpu_eval_batch_size 16 \
                              --evaluate_during_training \
                              --fp16 
~~~
If apex is not installed remove the  `--fp16`  parammeter

You can download pre-trained model from the links mentioned above (layoutLM or drive)

Similarly, you can do evaluation by changing `--do_train` to `--do_eval` and `--do_test`

Like the sequence labeling task, you can run Bert and RoBERTa baseline by modifying the `--model_type` argument.

### Results

#### SROIE


| Model                                                        | Hmean      |
| ------------------------------------------------------------ | ---------- |
| BERT-Large                                                   | 90.99%     |
| RoBERTa-Large                                                | 92.80%     |
| [Ranking 1st in SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) | 94.02%     |
| [**LayoutLM**](https://rrc.cvc.uab.es/?ch=13&com=evaluation&view=method_info&task=3&m=71448) | **96.04%** |

#### RVL-CDIP

| Model                                                        | Accuracy   |
| ------------------------------------------------------------ | ---------- |
| BERT-Large                                                   | 89.92%     |
| RoBERTa-Large                                                | 90.11%     |
| [VGG-16 (Afzal et al., 2017)](https://arxiv.org/abs/1704.03557) | 90.97%     |
| [Stacked CNN Ensemble (Das et al., 2018)](https://arxiv.org/abs/1801.09321) | 92.21%     |
| [LadderNet (Sarkhel & Nandi, 2019)](https://www.ijcai.org/Proceedings/2019/0466.pdf) | 92.77%     |
| [Multimodal Ensemble (Dauphinee et al., 2019)](https://arxiv.org/abs/1912.04376) | 93.07%     |
| **LayoutLM**                                                 | **94.42%** |

#### FUNSD

| Model         | Precision  | Recall     | F1         |
| ------------- | ---------- | ---------- | ---------- |
| BERT-Large    | 0.6113     | 0.7085     | 0.6563     |
| RoBERTa-Large | 0.6780     | 0.7391     | 0.7072     |
| **LayoutLM**  | **0.7677** | **0.8195** | **0.7927** |

## Citation

If you find LayoutLM useful in your research, please cite the following paper:

``` latex
@misc{xu2019layoutlm,
    title={LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
    author={Yiheng Xu and Minghao Li and Lei Cui and Shaohan Huang and Furu Wei and Ming Zhou},
    year={2019},
    eprint={1912.13318},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using LayoutLM, please submit a GitHub issue.

For other communications related to LayoutLM, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

