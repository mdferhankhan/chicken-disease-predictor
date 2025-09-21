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

def home(request):
    result = None
    confidence = None
    image_url = None

    if request.method == "POST" and request.FILES.get("image"):
        img_file = request.FILES["image"]

        # Remove old images from media folder
        media_folder = settings.MEDIA_ROOT
        for filename in os.listdir(media_folder):
            file_path = os.path.join(media_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

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

        # Store in session
        request.session['result'] = result
        request.session['confidence'] = confidence
        request.session['image_url'] = image_url

        return redirect("/")

    # Get from session if available
    result = request.session.pop('result', None)
    confidence = request.session.pop('confidence', None)
    image_url = request.session.pop('image_url', None)

    return render(request, "home.html", {
        "result": result,
        "confidence": confidence,
        "image_url": image_url
    })
