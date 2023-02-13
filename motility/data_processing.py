import os
import os.path as op
from pathlib   import Path
from glob      import glob
from datetime  import datetime

import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ['t', 'x', 'y']
FILE_PATH = "data/test_features.csv"

def data_to_df(dict, labels):

    out = {}
    for d in dict.keys():
        out[d] = pd.DataFrame(dict[d], columns=labels)
    return out


def get_uid_to_data(file_path):

    uids = [g.removeprefix(file_path + "/").removesuffix(".npy") for g in glob(f'{file_path}/*.npy')]
    return { u : np.load(file_path + "/" + u + ".npy") for u in uids}


def features_to_csv(file_path, dict, features_list):
    dict.DataFrame.from_dict(dict, orient='index').to_csv(file_path, headerbool=features_list)


# data = get_uid_to_data('data/test_npys')
# print(data)
# df = data_to_df(data, LABELS)
# print(df)
