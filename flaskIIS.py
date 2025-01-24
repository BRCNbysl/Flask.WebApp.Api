from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import logging
import importlib

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}, r"/feedback": {"origins": "*"}})

@app.route('/')
def hello_world():
    return 'Welcome to Mubitek Artificial Intelligence Solutions'

# Önceden eğitilmiş modeli yükle
model = joblib.load("cat_boost_model_updated.pkl")

# Logging ayarları
logging.basicConfig(filename='api_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # İstekten form verilerini al
        data = request.form
        logging.info(f"Received predict request data: {data}")

        # Gelen verileri modele uygun formata dönüştür
        pattern_name = float(data["KalıpAdı"])
        length = float(data["En"])
        width = float(data["Boy"])
        height = float(data["Yükseklik"])
        coefficient = float(data["Katsayı"])

        # Gelen verileri bir NumPy dizisine dönüştür
        additional_info = np.array([[pattern_name, length, width, height, coefficient]])

        # Modelin tahminini yap
        pred = model.predict(additional_info.reshape(1, -1))[0]

        # Tahmini JSON formatında döndür
        # Formatla ve yalnızca virgülden sonra ikinci basamağa kadar al
        formatted_pred = "{:.2f}".format(pred)
        response = {"success": True, "prediction": formatted_pred}
        logging.info(f"Prediction response: {response}")
        return jsonify(response)

    except Exception as e:
        # Hata durumunda hata mesajını döndür
        error_response = {"success": False, "error": str(e)}
        logging.error(f"Error in predict endpoint: {error_response}")
        return jsonify(error_response)

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        # model_vs2'yi dinamik olarak içe aktar
        model_vs2 = importlib.import_module('model_vs2')

        data = request.json
        logging.info(f"Received feedback request data: {data}")
        correct_target = float(data["DolulukOrani"])

        # Örnek olarak X_train ve y_train değişkenlerine erişelim
        X_train = model_vs2.X_train
        y_train = model_vs2.y_train

        # Yeni girdiyi DataFrame'e dönüştür
        new_data_point = np.array([[float(data["KalıpAdı"]), float(data["En"]), float(data["Boy"]), float(data["Yükseklik"]), float(data["Katsayı"]), correct_target]])

        # Yeni girdiyi eğitim veri setine ekleyerek modeli güncelle
        X_train_updated = np.vstack([X_train, new_data_point[:,:-1]])
        y_train_updated = np.concatenate([y_train, new_data_point[:,-1]])
        model.fit(X_train_updated, y_train_updated)
        logging.info("Model Güncellendi")

        # Yeniden eğitilmiş modeli pkl dosyasına kaydet
        joblib.dump(model, "cat_boost_model_updated.pkl")

        # Güncellenmiş modelin tahmini sonucu JSON formatında döndür
        updated_pred = model.predict(new_data_point[:,:-1])[0]
        response = {"success": True, "updated_prediction": updated_pred}
        logging.info(f"Feedback response: {response}")
        return jsonify(response)

    except Exception as e:
        # Hata durumunda hata mesajını döndür
        error_response = {"success": False, "error": str(e)}
        logging.error(f"Error in feedback endpoint: {error_response}")
        return jsonify(error_response)

if __name__ == '__main__':
    app.run(debug=True)
