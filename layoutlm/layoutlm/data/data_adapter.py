import os, uuid, base64, sys
import re 

class DataAdapter():

    DATA_DIR = 'data'

    def __init__(self):
        self.data_dir = self.DATA_DIR    
        self.label_dir = os.path.join(self.DATA_DIR, 'labels')
        self.xml_dir = os.path.join(self.DATA_DIR, 'images')
        self.tiff_dir = os.path.join(self.DATA_DIR, 'tiffs')
        self.version_file = os.path.join(self.label_dir, "version.txt")

    def set_xml_path(self, template_id): 
        return os.path.join(self.xml_dir,template_id)

    def set_img_path(self, template_id): 
        return os.path.join(self.tiff_dir,template_id)

    def set_label_path(self):
        return os.path.join(self.label_dir,'train.txt')

    def add_img_dir(self, template_id, filename):
        img_path= self.set_img_path(template_id)
        try:
            os.mkdir(img_path)
        except:
            pass
        img = os.path.join(img_path, f'{filename}.tiff')
        return img

    def add_img (self, img, base64_img):
        with open(img, 'wb') as file_to_save:
            decoded_image_data = base64.b64decode(base64_img, '-_')
            file_to_save.write(decoded_image_data)

    def add_xml_to_dir(self, template_id, filename):
        xml_path = self.set_xml_path(template_id)
        try:
            os.mkdir(xml_path)
        except:
            pass
        xml_file = os.path.join(xml_path,f'{filename}.xml')
        return xml_file

    def set_filepath(self, template_id, filename):
        return f'{template_id}/{filename}.xml'

    def write_training_label(self, filepath, label):
        training_labels_file= self.set_label_path()
        with open(training_labels_file, 'a+') as file_object:
            file_object.write('\n')
            file_object.write(f'{filepath} {label}')

    def get_data_version(self):
        f = open(self.version_file, "r")
        text=f.read()
        return text
    
    def write_updated_version(self, text):
        f = open(self.version_file, "w")
        f.write(text)

    def find_URL(self, string): 
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        url = re.findall(regex,string)
        return [x[0] for x in url]

    def adapt_data(self, path):
        new_path = path

        # check if it's a directory
        if(os.path.isdir(path)):
            print("is directory")
            return new_path

        # check if it's a link
        elif(len(self.find_URL(path)) != 0 ):
            print("islink")
            # connect to a DB using the link and "load" collection into new_path
            # new_path = collection
            return new_path
        
        # if it's not 
        else:
            return {"error": "wrong path"}

    