import numpy as np

def tensor_value_info(x, name=None):
    if name is not None:  
        print(f"tensor {name} infomation < shape: {x.shape} dtype: {x.dtype} min: {np.min(x):.3f} max: {np.max(x):.3f} mean: {np.mean(x):.3f} >")
    else:
        print(f"< shape: {x.shape} dtype: {x.dtype} min: {np.min(x):.3f} max: {np.max(x):.3f} mean: {np.mean(x):.3f} >")
