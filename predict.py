import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()
import json
import glob
from PIL import Image
import argparse

class ImageClassifier:
    
    def __init__(self, files, model_path, top_k, class_names_path):
        self.files = glob.glob(files)
        self.model_path = model_path
        self.top_k = int(top_k)
        self.class_names_path = class_names_path

        self.load_class_names()
        self.load_model()
        
        self.prababilities = []
        self.top_classes = []
        
        for img_pth in self.files:
            im = Image.open(img_pth)
            self.image_array = np.asarray(im)
            self.process_image()
            probs, classes= self.predict()
            
            # In case we want to export them
            self.prababilities.append(probs)
            self.top_classes.append(classes)
        
    def load_class_names(self):
        with open(self.class_names_path, 'r') as f:
            self.class_names = json.load(f)
           
    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path, custom_objects={'KerasLayer':hub.KerasLayer})
        #print(self.model.summary())

    def process_image(self):
        image = np.squeeze(self.image_array)
        image = tf.image.resize(image, (224, 224))   
        self.processed_test_image = (image/255)
    
    def predict(self):
        processed_test_image = np.expand_dims(self.processed_test_image, axis=0)
        ps = self.model.predict(processed_test_image)
        top_val, indx = tf.math.top_k(ps, self.top_k)
        indx += 1
        print("\nTop propabilities {}".format(top_val.numpy()[0]))
        top_classes = [self.class_names[str(value)] for value in indx.numpy()[0]]
        print('Top classes {}'.format(top_classes))
        return top_val.numpy()[0], top_classes
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for parser")
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("load_model",help="Model Path", default="")
    parser.add_argument("--top_k", help="Return the top K most likely classes", required = False, default = 3)
    parser.add_argument("--category_names", help=" Path to a JSON file mapping labels to flower names", required = False, default = "label_map.json")
    args = parser.parse_args()

    ImageClassifier(args.image_path, args.load_model, args.top_k, args.category_names)
    
#python predict.py './test_images/*.jpg' 'my_model.h5 --top_k "3" --category_names label_map.json