from utils import Utils
from models import Models

if __name__ == "__main__":

    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./data/DatosDistribuidora.csv')
    X, y = utils.features_target(data, ['target'],['target'])

    models.training(X,y)

    print(data)

    #creacion_exportacion_modelo