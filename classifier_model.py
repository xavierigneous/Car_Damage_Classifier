from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import VGG16
import os, base64
import numpy as np
damage=['Damaged', 'No Damage']
location=['Front','Rear','Side']
level = ['Minor','Moderate','Severe']
MODEL_FOLDER = os.path.join(os.path.dirname(__file__),'models')
def image_preprocess(filename):
    image = load_img(filename, target_size=(300, 300))
    image = np.expand_dims(img_to_array(image) / 255, axis=0)
    return(image)

def image_preprocess_small(filename):
    image = load_img(filename, target_size=(224, 224))
    image = np.expand_dims(img_to_array(image) / 255, axis=0)
    return(image)

def return_image(uploaded_image):
    display_image = base64.b64encode(uploaded_image.read()).decode()
    return display_image

def get_prediction(image):
    damage_model = load_model(os.path.join(MODEL_FOLDER,'MobileNet_Car_Classifier.h5'))
    label = damage[int(damage_model.predict(image).argmax(axis=-1))]
    print(label)
    return(label)

def get_damage_part(image):
    damage_part_model = load_model(os.path.join(MODEL_FOLDER,'MobileNet_Car_Damaged_Part.h5'), compile=False)
    label = location[int(damage_part_model.predict(image).argmax(axis=-1))]
    print(label)
    return(label)

def get_damage_severity(image):
    damage_part_model = load_model(os.path.join(MODEL_FOLDER,'MobileNet_Damage_Level_Classifier.h5'), compile=False)
    label = level[int(damage_part_model.predict(image).argmax(axis=-1))]
    print(label)
    return(label)