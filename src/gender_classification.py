import sys
import os
import cv2
from keras.models import load_model
import numpy as np
from tqdm import tqdm
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_image
from utils.preprocessor import preprocess_input


if __name__ == "__main__":
    INPUT_FOLDER = '/home/oem/project/Face_Expression/3.Data/wiki_500'
    gender_model_path = '/home/oem/project/Face_Expression/8.face_gender_emotion/trained_models/gender_models/simple_CNN.81-0.96.hdf5'
    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')

    # hyper-parameters for bounding boxes shape
    gender_offsets = (10, 10)
    gender_classifier = load_model(gender_model_path, compile=False)
    gender_target_size = gender_classifier.input_shape[1:3]

    count = 0
    total = 0
    for label_dir in os.listdir(INPUT_FOLDER):
        label_dir_path = os.path.join(INPUT_FOLDER, label_dir)
        for img_name in tqdm(os.listdir(label_dir_path)):
            # loading images
            img_path = os.path.join(label_dir_path, img_name)
            rgb_image = load_image(img_path, grayscale=False)
            rgb_face = cv2.resize(rgb_image, (gender_target_size))
            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            
            total+=1

            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]
            if (gender_text == label_dir):
                count+=1
    
    acc = (count/total) * 100
    print(str(acc) + '%')

