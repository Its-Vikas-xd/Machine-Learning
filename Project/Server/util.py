import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    """
    Estimate home price given location, sqft, BHK, and bath.
    If location is not in the dataset, it will be treated as 'other'.
    """
    if location:
        location = location.strip().lower()
    
    try:
        # find index of location in data columns (case-insensitive)
        loc_index = next(i for i, col in enumerate(__data_columns[3:]) 
                         if col.lower() == location)
        loc_index += 3  # offset by first 3 numeric features
    except StopIteration:
        loc_index = -1  # unknown location

    # initialize feature array
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    # predict price using the trained model
    estimated_price = round(__model.predict([x])[0], 2)
    return estimated_price


def load_saved_artifacts():
    """
    Load trained model and column information from artifacts folder.
    """
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Load column info
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are numeric features

    # Load trained model
    if __model is None:
        with open('./artifacts/banglor_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)

    print("Loading saved artifacts...done")


def get_location_names():
    """
    Returns list of all locations from the dataset.
    """
    return __locations


def get_data_columns():
    """
    Returns all feature columns (including numeric and location columns)
    """
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()

    print("Available locations:", get_location_names())
    
    # Test examples
    print("Estimated price:", get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print("Estimated price:", get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print("Estimated price:", get_estimated_price('Kalhalli', 1000, 2, 2))   # unseen location
    print("Estimated price:", get_estimated_price('Ejipura', 1000, 2, 2))    # unseen location
