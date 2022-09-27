import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([15,7.00,9,2.00,0,1,1,1,1,1,1,1,1,3,1,2,1,1,1,1,1,0,1,1,1,2])
    prediction = model.predict(X_test.reshape(1,-1))

    return jsonify({'prediccion' : list(prediction)})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8088)