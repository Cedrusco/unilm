import os, sys, subprocess, shutil

# This script can be run to pass each image within a dataset through tesseract-OCR to generate HOCR format .xml files
# Navigate to the directory of this file and run command "python3 convert.py path/to/dataset/"
# Also provides method convert_img_to_xml to be used in prediction and training scripts

OUT_DIR = '/out-OCR/'

# Convert an entire directory of images and generate .xml files in new directory ~/out-OCR/
# Expects one argument, root - string path to image directory
def convert_all_to_xml(root):
    output_dir = root + OUT_DIR
    try:
        os.mkdir(output_dir) # make output directory if it does not exist already
    except:
        pass
    for path, directories, files in os.walk(root):
        for file in files:  # walk through dataset for img files
            if file.endswith('.png') | file.endswith('.jpg') | file.endswith('.tiff') | file.endswith('.tif'):
                img = os.path.join(path, file)
                convert_img_to_xml(img, output_dir, os.path.splitext(file)[0]) # convert single img


# Convert a single image to HOCR format .xml file
# Expects three arguments, img - path to image to convert,
# output_dir - directory to generate output files,
# filename - name of output file
def convert_img_to_xml(img, output_dir, filename):
    output_file = os.path.join(output_dir, filename)
    subprocess.run(['tesseract', img, output_file, 'hocr']) # make call to Tesseract with img file and output dir
    old_file_path = output_file + '.hocr'
    new_file_path = output_file + '.xml'
    if not os.path.exists(new_file_path):
        shutil.copy(old_file_path, new_file_path) # copy .hocr file contents into .xml file
        os.remove(old_file_path) # delete original .hocr file


if __name__ == "__main__":
    convert_all_to_xml(sys.argv[1])
