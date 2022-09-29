import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.json
    x = list(datos.values())
    # X_test = np.reshape(x, (1,-1))
    # prediction = model.predict(X_test)
    #return jsonify({'prediccion' : list(prediction)})
    
    print(datos.values())
    query_df = pd.DataFrame(x[0])
    print(query_df.shape)
    print(query_df.dtypes)
    print(query_df.head())

    #X_test = np.reshape(query_df, (1,-1))
    prediction = model.predict(query_df)
    return jsonify({'prediccion' : list(prediction)})


if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8888)