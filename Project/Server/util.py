import json
import pickle

__location = None
__data_columns = None
__model = None

def get_location_names():
    """Return list of locations"""
    return __location

def load_saved_artifacts():
    print("loading saved artifacts....start")
    global __data_columns
    global __location
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)["data_columns"]
        __location = __data_columns[3:]   # first 3 are sqft, bath, bhk

    with open("./artifacts/banglor_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts ....done")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
