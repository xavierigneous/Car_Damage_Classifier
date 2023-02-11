from flask import Flask, request, jsonify, render_template
import keras
from classifier_model import *
import json, base64, os, time, sys

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__),'uploads')
print(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

        damage_part = get_damage_part(image_data)

        # uploaded_image = file #return_image(file)

        # Return the prediction result as JSON
        return render_template('homepage.html', 
        damage_report=prediction, 
        damage_part=damage_part,
        uploaded_image = filename)
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(debug=True)