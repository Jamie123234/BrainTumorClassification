import os
import joblib
import numpy as np
from flask import Flask, request, render_template, flash
from skimage import io, transform
from werkzeug.utils import secure_filename # For handling filenames securely

# --- Configuration ---
MODEL_FILENAME = 'brain_tumor_classifier.joblib'
UPLOAD_FOLDER = 'uploads' # Create this folder inside 'BrainTumorClassification'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image parameters MUST match those used during training
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
EXPECTED_FLAT_LEN = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key' # Needed for flashing messages

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Load the Pre-trained Model ---
try:
    model = joblib.load(MODEL_FILENAME)
    print(f"Model '{MODEL_FILENAME}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILENAME}' not found.")
    model = None # Handle cases where model loading fails
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Loads and preprocesses an image file like in training."""
    try:
        img = io.imread(image_path, as_gray=True)

        img_resized = transform.resize(img, IMG_SIZE, anti_aliasing=True)
        img_flattened = img_resized.flatten()

        return [img_flattened]

    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, prediction, and displays result."""
    if model is None:
        flash('Model is not loaded. Cannot make predictions.', 'error')
        return render_template('index.html', prediction=None)

    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return render_template('index.html', prediction=None)

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        flash('No selected file.', 'warning')
        return render_template('index.html', prediction=None)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            print(f"File saved to {filepath}")

            # Preprocess the uploaded image
            processed_image = preprocess_image(filepath)

            if processed_image is not None:
                # Make prediction
                prediction_code = model.predict(processed_image)[0] # Get the single prediction
                prediction_proba = model.predict_proba(processed_image)[0] # Get probabilities [P(class 0), P(class 1)]

                # Interpret prediction
                if prediction_code == 1:
                    result_text = "Tumor Detected"
                    confidence = prediction_proba[1] * 100 # Confidence in 'Tumor'
                else:
                    result_text = "No Tumor Detected"
                    confidence = prediction_proba[0] * 100 # Confidence in 'No Tumor'

                prediction_result = f"{result_text} (Confidence: {confidence:.2f}%)"
                print(f"Prediction: {prediction_result}")
                if (result_text == "No Tumor Detected"):
                    flash(f"Prediction complete: {prediction_result}", 'success')
                else:
                    flash(f"Prediction complete: {prediction_result}", 'error')

                # Clean up the uploaded file (optional)
                # os.remove(filepath)
                return render_template('index.html', prediction=prediction_result, filename=filename)

            else:
                flash('Error processing the uploaded image.', 'error')
                # os.remove(filepath) # Clean up failed file
                return render_template('index.html', prediction=None)

        except Exception as e:
            flash(f'An error occurred: {e}', 'error')
            # Ensure file is removed if it exists and error occurred
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('index.html', prediction=None)

    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg.', 'error')
        return render_template('index.html', prediction=None)

# --- Run the App (for local testing) ---
if __name__ == '__main__':
    # Make sure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # Run the app (accessible only on your computer)
    # Use host='0.0.0.0' to make it accessible on your local network (e.g., from your phone)
    app.run(debug=True, host='0.0.0.0')