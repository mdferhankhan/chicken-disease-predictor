# Chicken Disease Predictor

This project is a Django web app that allows users to upload chicken images and predicts diseases using a trained CNN model (EfficientNetB3).

## Features
- Upload chicken images via UI
- Predict disease class (Coccidiosis, Salmonella, Newcastle Disease, Healthy)
- Model trained on dataset with ~98% accuracy

## Setup
```bash
git clone https://github.com/mdferhankhan/chicken-disease-predictor.git
cd chicken-disease-predictor
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
