# Crop Disease Detection

 
## Overview
This project is a Flask-based crop disease detection web application that allows users to upload leaf images and get disease predictions using a trained TensorFlow/Keras model. It also includes a weather forecast section that uses the user's location or a manually entered city name to show local weather conditions.

## Dataset
The disease detection model is trained on a crop disease image dataset containing multiple classes of healthy and diseased plant leaves. The label-to-disease mapping is stored in `plant_disease.json`, and the trained model file is stored in `models/crop_disease_model.keras`.

## Data Preprocessing
Before prediction, uploaded images are:
1. Loaded using TensorFlow image utilities.
2. Resized to `160 x 160` pixels.
3. Converted into a NumPy array.
4. Expanded into a batch dimension for model inference.

This preprocessing ensures the image format matches the model input requirements.

## Exploratory Data Analysis
The project focuses on visual inspection of crop disease images and class label interpretation. The dataset is organized into disease categories, and the model output is mapped back to human-readable descriptions from `plant_disease.json`.

Key analysis aspects include:
- checking image quality and resolution,
- verifying disease class names,
- confirming that the prediction labels are correctly mapped to explanations and cure suggestions.

## Model Training
The model is built and trained using TensorFlow/Keras. The prediction pipeline loads the trained model from `models/crop_disease_model.keras` and uses it to classify uploaded crop leaf images.

The main prediction flow is:
- upload image through the Flask app,
- preprocess image,
- pass the image to the Keras model,
- display the predicted disease name, cause, and cure.

## Results
The application provides:
- crop disease prediction output with disease details,
- explanatory text from the JSON label file,
- a weather card that shows the current weather using location-based or city-based lookup.

The result page displays the predicted disease along with guidance for treatment and prevention.

## How to Run
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask app:
   ```bash
   python app.py
   ```
4. Open the app in your browser at:
   ```text
   http://127.0.0.1:5000/
   ```

## Future Improvements
Possible improvements for the project include:
- adding more crop disease classes,
- improving model accuracy with a larger dataset,
- adding support for multiple languages,
- integrating a more advanced weather API or forecast details,
- deploying the app to a cloud platform such as Render, Railway, or Vercel.
