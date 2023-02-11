from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import VGG16
import app, os, base64
import numpy as np
damage=['Damage', 'No Damage']
location=['Front','Rear','Side']
MODEL_FOLDER = os.path.join(os.path.dirname(__file__),'models')
def image_preprocess(filename):
    image = load_img(filename, target_size=(300, 300))
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