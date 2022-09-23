from utils import Utils
from models import Models

if __name__ == "__main__":

    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./data/DatosDistribuidora.csv')
    #X, y = utils.features_target(data, ['CodProducto', 'Producto', 'Categoria', 'Activo', 'F_vencimiento', 'P_compra', 'P_venta', 'Margen' ,'target'],['target'])
    X, y = utils.features_target(data, ['target'],['target'])

    models.grid_training(X,y)

    print(data)

    #creacion_exportacion_modelo