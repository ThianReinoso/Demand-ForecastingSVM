import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([150,2.00,3.1,1.10,6,4,6,2,4,3,0,6,4,6,2,4,5,2,1,4,6,2,4,4,1,3,3,4,2,3,3,4,1,3,3,1,2,5,1,6,4,6,4,1,6,4,5,2,1,4,6,2,4,4,6,4])
    prediction = model.predict(X_test.reshape(1,-1))

    return jsonify({'prediccion' : list(prediction)})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8088)