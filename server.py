import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([15,8.00,10,2.00,8,2,9,3,6,5,3,5,7,6,5,8,4,5,8,4,5,8,3,5,5,5])
    prediction = model.predict(X_test.reshape(1,-1))

    return jsonify({'prediccion' : list(prediction)})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8088)