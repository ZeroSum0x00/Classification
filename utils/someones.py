import sys
import pickle
from copy import deepcopy
import numpy as np

def exit():
    sys.exit()
    
def save_pickle(file_path, data):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)


def load_pickle(file_pickle):
    with open(file_pickle, 'rb') as handle:
        data = pickle.load(handle)
    return data