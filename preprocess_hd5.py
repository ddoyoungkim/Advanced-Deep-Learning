import pandas as pd
import pathlib
import ast
from math import sin, ceil
import numpy as np

import data_utils as utils


data_dir = pathlib.PosixPath("data/")

utils.porto2h5(data_dir/"porto"/"train.csv", limit=None,
               fname=data_dir/"porto"/"preprocessed_entire_porto_in_py.h5",
               is_first=False)


# utils.porto2standardcsv(data_dir/"porto"/"train.csv")