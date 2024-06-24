from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import requests

app = FastAPI()

# Define the path to the CSV file
CSV_FILE_URL = 'https://firebasestorage.googleapis.com/v0/b/smart-agriculture-6c46a.appspot.com/o/Crop_recommendation.csv?alt=media&token=f9a87784-3298-4328-b8ab-b2be598a1615'

class DataModel(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

def load_data(file_url):
    response = requests.get(file_url)
    with open('Crop_recommendation.csv', 'wb') as file:
        file.write(response.content)
    
    df = pd.read_csv('Crop_recommendation.csv')
    X = df.drop('label', axis=1)
    y = df['label']

    sc = StandardScaler()
    X = sc.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, sc, le

def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_train, y_train):
    return cross_val_score(model, X_train, y_train, cv=10).mean()

def save_model(model, scaler, label_encoder, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'label_encoder': label_encoder}, f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['label_encoder']

def make_prediction(model, scaler, label_encoder, new_data):
    new_data_df = pd.DataFrame([new_data.dict()])
    new_data_df = scaler.transform(new_data_df)
    prediction = model.predict(new_data_df)
    prediction = label_encoder.inverse_transform(prediction)
    return prediction[0]

@app.post('/train')
def train():
    X_train, X_test, y_train, y_test, sc, le = load_data(CSV_FILE_URL)
    model = train_model(X_train, y_train)
    model_score = evaluate_model(model, X_train, y_train)
    save_model(model, sc, le, 'crop_model.pkl')
    return {'message': 'Model trained and saved successfully', 'model_score': model_score}

@app.post('/predict')
def predict(data: DataModel):
    model, scaler, label_encoder = load_model('crop_model.pkl')
    prediction = make_prediction(model, scaler, label_encoder, data)
    return {'predicted_crop': prediction}

# Run the server with: uvicorn main:app --reload
