import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from utils import Utils

class Models:

    def __init__(self):
        self.reg = {
            'Regression' : LinearRegression(),
        }

    def training(self, X,y):

        StandardScaler().fit_transform(X)

        # Partimos el conjunto de entrenamiento. Para añadir replicabilidad usamos el random state
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ahora vamos a configurar nuestra regresión logística
        regression = LinearRegression()
    
        # Mandamos los data frames la la regresión logística
        regression.fit(X_train, y_train)

        score = regression.score(X_train, y_train)

        utils = Utils()
        utils.model_export(LinearRegression, score)