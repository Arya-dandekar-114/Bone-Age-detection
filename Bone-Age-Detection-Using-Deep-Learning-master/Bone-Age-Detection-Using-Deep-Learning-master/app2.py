from flask import Flask, render_template_string, request
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Model path
MODEL_PATH = r'C:\Users\aryad\Downloads\Bone-Age-Detection-Using-Deep-Learning-master\Bone-Age-Detection-Using-Deep-Learning-master\Web App Development\model.h5'

# Load the model once to avoid multiple loads
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# HTML template for the index page
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bone Age Predictor</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mt-5">Bone Age Prediction</h1>
        <form method="POST" enctype="multipart/form-data" class="mt-3">
            <div class="form-group">
                <label for="image">Upload Hand X-Ray Image</label>
                <input type="file" name="image" class="form-control-file" required>
            </div>
            <div class="form-group">
                <label for="gender">Select Gender</label>
                <select name="gender" class="form-control" required>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="unknown">Unknown</option>
                </select>
            </div>
            <button type="submit" class="btn btn-primary">Predict</button>
        </form>
        {% if error %}
            <div class="alert alert-danger mt-3">{{ error }}</div>
        {% endif %}
    </div>
</body>
</html>
'''

# HTML template for displaying the prediction result
PREDICTION_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mt-5">Prediction Result</h1>
        <p class="mt-3">Predicted Bone Age: {{ prediction }} years</p>
        <a href="{{ url_for('index') }}" class="btn btn-primary mt-3">Back</a>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure a file is uploaded
        if 'image' not in request.files:
            return render_template_string(INDEX_HTML, error="No file part")
        img = request.files['image']
        if img.filename == '':
            return render_template_string(INDEX_HTML, error="No selected file")
        
        # Save the image with a secure filename
        img_dir = 'static/' + secure_filename(img.filename)
        img.save(img_dir)
        
        try:
            # Preprocess the image
            img = tf.keras.preprocessing.image.load_img(img_dir, target_size=(256, 256))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.xception.preprocess_input(img)
            
            # Predict bone age
            mean_bone_age = 127.3207517246848
            std_bone_age = 41.18202139939618
            pred = round((mean_bone_age + std_bone_age * (model.predict(np.array([img]))[0][0])) / 12, 2)
            
            return render_template_string(PREDICTION_HTML, prediction=pred)
        except Exception as e:
            return render_template_string(INDEX_HTML, error=f"Error in processing image: {e}")
    return render_template_string(INDEX_HTML)

if __name__ == '__main__':
    # Ensure the static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
