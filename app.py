from flask import Flask, request, jsonify, render_template
import keras
from classifier_model import *
import json, base64, os, time, sys

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__),'uploads')
print(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load the saved TensorFlow model
# model = tf.keras.models.load_model('model.h5')

# damage_model = keras.models.load_model(r'classifier\models\MobileNet_Car_Classifier.h5')
# location_pred_model = keras.models.load_model(r'classifier\models\MobileNet_Car_Damage_Location.h5')

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(filename)

        # Read the image data
        image_data = image_preprocess(filename)

        # Use the TensorFlow model to make a prediction
        prediction = get_prediction(image_data)

        # uploaded_image = file #return_image(file)

        # Return the prediction result as JSON
        return render_template('homepage.html', damage_report=prediction, uploaded_image = filename)
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(debug=True)
