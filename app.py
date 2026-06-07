from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import requests

API_KEY = "bfd851a6ce85aad45ccc612b9321aecf"

app = Flask(__name__)

model = tf.keras.models.load_model("models/crop_disease_model.keras")

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature


def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image.save(f'{temp_name}_{image.filename}')
        prediction = model_predict(f'./{temp_name}_{image.filename}')
        return render_template('home.html', result=True,
                               imagepath=f'/{temp_name}_{image.filename}',
                               prediction=prediction)
    else:
        return redirect('/')


@app.route('/weather')
def weather():
    lat = request.args.get("lat", "").strip()
    lon = request.args.get("lon", "").strip()
    city = request.args.get("city", "").strip()

    if lat and lon:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    elif city:
        geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_resp = requests.get(geo_url, timeout=15)
        geo_data = geo_resp.json()

        if not geo_data:
            return {"error": True, "message": "City not found"}, 404

        lat = geo_data[0].get("lat")
        lon = geo_data[0].get("lon")
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    else:
        return {"error": True, "message": "Missing latitude, longitude, or city"}, 400

    response = requests.get(weather_url, timeout=15)
    data = response.json()

    if response.status_code != 200 or "main" not in data:
        return {
            "error": True,
            "message": data.get("message", "Weather API failed"),
            "raw": data
        }, response.status_code if response.status_code != 200 else 500

    return {
        "error": False,
        "location": data.get("name", "Unknown"),
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"],
        "description": data["weather"][0]["description"]
    }

if __name__ == "__main__":
    app.run(debug=True)
