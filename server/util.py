import json
import pickle
import numpy as np


__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __data_columns
    global __locations

    with open("server/artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[5:]

    global __model
    with open("server/artifacts/housing_price_prediction.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("saved artifacts loaded.")



if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    