# # Authors: Pratik Poojary, Prathamesh Pokhare

# from flask import Flask, render_template, request, url_for
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# import numpy as np
# import os

# app = Flask(__name__)

# # Model path (using relative path)
# MODEL_PATH = MODEL_PATH = r'C:/Users/aryad/Downloads/Bone-Age-Detection-Using-Deep-Learning-master/Bone-Age-Detection-Using-Deep-Learning-master/Web App Development/model.h5'


# # Load the model once to avoid multiple loads (if it's not too heavy)
# try:
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")

# @app.route('/')
# def main():
#     return render_template('index.html')


# @app.route('/index/', methods=['GET', 'POST'])
# def index():
#     # Ensure a file is uploaded
#     if 'image' not in request.files:
#         return render_template('index.html', error="No file part")

#     img = request.files['image']

#     if img.filename == '':
#         return render_template('index.html', error="No selected file")

#     # Save the image with a secure filename
#     img_dir = 'static/' + secure_filename(img.filename)
#     img.save(img_dir)

#     try:
#         # Preprocess the image
#         img = tf.keras.preprocessing.image.load_img(img_dir, target_size=(256, 256))
#         img = tf.keras.preprocessing.image.img_to_array(img)
#         img = tf.keras.applications.xception.preprocess_input(img)

#         # Normalize the bone age prediction
#         mean_bone_age = 127.3207517246848
#         std_bone_age = 41.18202139939618
#         pred = round((mean_bone_age + std_bone_age * (model.predict(np.array([img]))[0][0])) / 12, 2)

#         final = [str(pred), img_dir]

#         return render_template('prediction.html', data=final)
#     except Exception as e:
#         return render_template('index.html', error=f"Error in processing image: {e}")


# @app.route('/about/')
# def about():
#     return render_template('about.html')


# if __name__ == '__main__':
#     app.run(debug=True)
# Authors: Pratik Poojary, Prathamesh Pokhare





















# from flask import Flask, render_template, request
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# import numpy as np
# import os

# app = Flask(__name__)

# # Model path (using relative path)
# MODEL_PATH = r'C:/Users/aryad/Downloads/Bone-Age-Detection-Using-Deep-Learning-master/Bone-Age-Detection-Using-Deep-Learning-master/Web App Development/model.h5'

# # Load the model once to avoid multiple loads (if it's not too heavy)
# try:
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")

# @app.route('/')
# def main():
#     return render_template('index.html')


# @app.route('/index/', methods=['GET', 'POST'])
# def index():
#     # Ensure a file is uploaded
#     if 'image' not in request.files:
#         return render_template('index.html', error="No file part")

#     img = request.files['image']

#     if img.filename == '':
#         return render_template('index.html', error="No selected file")

#     # Save the image with a secure filename
#     img_dir = 'static/' + secure_filename(img.filename)
#     img.save(img_dir)

#     try:
#         # Preprocess the image
#         img = tf.keras.preprocessing.image.load_img(img_dir, target_size=(256, 256))
#         img = tf.keras.preprocessing.image.img_to_array(img)
#         img = tf.keras.applications.xception.preprocess_input(img)

#         # Predict bone age
#         predicted_value = model.predict(np.array([img]))[0][0]
#         print(f"Prediction raw value: {predicted_value}")  # Debugging line

#         if predicted_value is not None:
#             mean_bone_age = 127.3207517246848
#             std_bone_age = 41.18202139939618
#             pred = round((mean_bone_age + std_bone_age * predicted_value) / 12, 2)
#         else:
#             pred = "Prediction error"

#         final = [str(pred), img_dir]

#         return render_template('prediction.html', data=final)
#     except Exception as e:
#         print(f"Error processing image: {e}")  # Debugging line
#         return render_template('index.html', error=f"Error in processing image: {e}")


# @app.route('/about/')
# def about():
#     return render_template('about.html')


# if __name__ == '__main__':
#     app.run(debug=True)







from flask import Flask, render_template, request
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        return
    try:
        # Placeholder architecture - replace with actual model from training code
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(256, 256, 3)),  # Matches your target_size
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)  # Single output for bone age
        ])
        model.load_weights(MODEL_PATH)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")

load_model()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/index/', methods=['GET', 'POST'])
def index():
    if 'image' not in request.files:
        return render_template('index.html', error="No file part")
    img = request.files['image']
    if img.filename == '':
        return render_template('index.html', error="No selected file")
    img_dir = os.path.join('static', secure_filename(img.filename))
    if not os.path.exists('static'):
        os.makedirs('static')
    img.save(img_dir)
    try:
        if model is None:
            return render_template('index.html', error="Model not loaded. Check server logs for details.")
        img = tf.keras.preprocessing.image.load_img(img_dir, target_size=(256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.xception.preprocess_input(img)
        predicted_value = model.predict(np.array([img]))[0][0]
        print(f"Prediction raw value: {predicted_value}")
        if predicted_value is not None:
            mean_bone_age = 127.3207517246848
            std_bone_age = 41.18202139939618
            pred = round((mean_bone_age + std_bone_age * predicted_value) / 12, 2)
        else:
            pred = "Prediction error"
        final = [str(pred), img_dir]
        return render_template('prediction.html', data=final)
    except Exception as e:
        print(f"Error processing image: {e}")
        return render_template('index.html', error=f"Error in processing image: {e}")

@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)