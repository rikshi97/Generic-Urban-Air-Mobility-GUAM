import pandas as pd
import scipy.io as spio


# export PYTHONPATH=$PYTHONPATH:'/Users/yilun/Desktop/REALM/jax_guam/'
def read_data(filename) -> dict:
    mat = spio.loadmat(filename)
    return mat


def read_data_dat(filename):
    df = pd.read_csv(filename, delimiter=" ")
    return df.to_numpy()
