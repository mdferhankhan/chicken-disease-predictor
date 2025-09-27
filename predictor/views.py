# Create your views here.
from django.shortcuts import render, redirect
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Load model and class dictionary once
MODEL_PATH = os.path.join(settings.BASE_DIR, "model_files", "Chicken Disease-97.65.h5")
CLASS_CSV_PATH = os.path.join(settings.BASE_DIR, "model_files", "Chicken Disease-class_dict.csv")

model = load_model(MODEL_PATH)
class_df = pd.read_csv(CLASS_CSV_PATH)
class_dict = dict(zip(class_df["class_index"], class_df["class"]))

def get_ideal_temperature(age):
    """
    Returns ideal broiler body temperature (°C) by age in days.
    Data based on Indian poultry production references (e.g., CARI, ICAR).
    """
    age = int(age)
    if age <= 7:
        return 40.5  # 0-7 days
    elif age <= 14:
        return 41.0  # 8-14 days
    elif age <= 21:
        return 41.2  # 15-21 days
    elif age <= 28:
        return 41.5  # 22-28 days
    else:
        return 41.7  # 29+ days

def home(request):
    result = None
    confidence = None
    image_url = None
    age = None
    temperature = None
    temp_status = None

    if request.method == "POST":
        age = request.POST.get("age")
        temperature = request.POST.get("temperature")
        excreta_toggle = request.POST.get("excreta_toggle")
        img_file = request.FILES.get("image") if excreta_toggle == "on" else None

        # Remove old images from media folder
        media_folder = settings.MEDIA_ROOT
        for filename in os.listdir(media_folder):
            file_path = os.path.join(media_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        image_url = None
        result = None
        confidence = None

        if img_file:
            fs = FileSystemStorage()
            filename = fs.save(img_file.name, img_file)
            img_path = fs.path(filename)

            # Preprocess
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array)
            class_index = np.argmax(preds, axis=1)[0]
            result = class_dict[class_index]
            confidence = f"{np.max(preds) * 100:.2f}%"
            image_url = fs.url(filename)

        # Determine temperature status
        temp_status = None
        if age and temperature:
            try:
                ideal_temp = get_ideal_temperature(age)
                actual_temp = float(temperature)
                if abs(actual_temp - ideal_temp) <= 0.3:
                    temp_status = "Normal"
                elif actual_temp > ideal_temp:
                    temp_status = "High"
                else:
                    temp_status = "Low"
            except Exception:
                temp_status = None

        # Store in session
        request.session['result'] = result
        request.session['confidence'] = confidence
        request.session['image_url'] = image_url
        request.session['age'] = age
        request.session['temperature'] = temperature
        request.session['temp_status'] = temp_status

        return redirect("/")

    # Get from session if available
    result = request.session.pop('result', None)
    confidence = request.session.pop('confidence', None)
    image_url = request.session.pop('image_url', None)
    age = request.session.pop('age', None)
    temperature = request.session.pop('temperature', None)
    temp_status = request.session.pop('temp_status', None)

    return render(request, "home.html", {
        "result": result,
        "confidence": confidence,
        "image_url": image_url,
        "age": age,
        "temperature": temperature,
        "temp_status": temp_status
    })
