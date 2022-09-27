# Importamos las bibliotecas generales

import pandas as pd
import sklearn
import matplotlib.pyplot as plt 

# Importamos los módulos específicos

from sklearn.decomposition import KernelPCA


from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
    dt_dist = pd.read_csv('data/DatosDistribuidora.csv')

    # Imprimimos un encabezado con los primeros 5 registros
    print(dt_dist.head(5))

    # Guardamos nuestro dataset sin la columna de target
    dt_features = dt_dist.drop(['target'], axis=1)

    # Este será nuestro dataset, pero sin la columna
    dt_target = dt_dist['target']

    # Normalizamos los datos
    dt_features = StandardScaler().fit_transform(dt_features)

    # Partimos el conjunto de entrenamiento. Para añadir replicabilidad usamos el random state
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.2, random_state=42)

    # Consultamos la fórmula para nuestra tabla
    print(X_train.shape)
    print(y_train.shape)

    kpca = KernelPCA(n_components=25, kernel='poly' )
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    regression = SVR(kernel='linear', epsilon=0.05, C=100)
    
    regression.fit(dt_train, y_train)
    print("SCORE KPCA: ", regression.score(dt_test, y_test))

    #kernels_y_kpca