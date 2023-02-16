from flask import Flask, request, render_template

import os
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__),'uploads')
print(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('homepage.html')
from classifier_model import *
@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(filename)

        # Read the image data
        image_data = image_preprocess(filename)
        image_data_small = image_preprocess_small(filename)

        # Use the TensorFlow model to make a prediction
        prediction = get_prediction(image_data)
        if prediction == 'No Damage':
            damage_part = ''

            damage_level = ''
        else:

            damage_part = get_damage_part(image_data_small)

            damage_level = get_damage_severity(image_data_small)

        # uploaded_image = file #return_image(file)

        # Return the prediction result as JSON
        return render_template('homepage.html',
        damage_report=prediction,
        damage_part=damage_part,
        damage_level=damage_level,
        uploaded_image = filename)
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(debug=True)