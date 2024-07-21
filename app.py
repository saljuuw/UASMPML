from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import os

app = Flask(__name__)

# URL untuk file model di cloud storage (Google Drive)
model_url = 'https://drive.google.com/uc?export=download&id=1aBFS3PkTY_4bA8zeJm_t5KaoNkYV-b_c'

def download_model(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    # Verifikasi ukuran file untuk memastikan unduhan berhasil
    if os.path.getsize(save_path) < 1e6:  # Misalnya, minimal 1MB
        raise ValueError("Unduhan model gagal atau file terlalu kecil.")

# Unduh model dan muat ke dalam aplikasi
try:
    download_model(model_url, 'random_forest_model.pkl')
    model = joblib.load('random_forest_model.pkl')
except Exception as e:
    print(f"Error downloading or loading the model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features'])
    prediction = model.predict(features)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
