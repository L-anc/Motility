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

def data_to_df(dict, labels=LABELS):
    '''
    Params:
    `dict` - a dictionary that maps uids to numpy arrays
    `labels` - column headers to be used in every data frame

    Output:
    `out` -  a dictionary that maps uids to numpy arrays
    '''
    out = {}
    for d in dict.keys():
        out[d] = pd.DataFrame(dict[d], columns=labels)
    return out


def get_uid_to_data(file_path='data/test_npys'):
    '''
    Params:
    `file_path` - the address to the folder 

    Output: a dictionary that maps uids to numpy arrays
    '''
    uids = [g.removeprefix(file_path + "/").removesuffix(".npy") for g in glob(f'{file_path}/*.npy')]
    return { u : np.load(file_path + "/" + u + ".npy") for u in uids}


def features_to_csv(file_path, dict, features_list):
    '''
    Params:
    `file_path` - path to the csv file to which data is being saved
    `dict` - dictionary that maps uids to the feature values. 
    `features list` - list of features that are used as the headers
    '''
    dict.DataFrame.from_dict(dict, orient='index').to_csv(file_path, headerbool=features_list)

def load_all_data():
    '''
    Loads all .csv data into .npy files to be used for easier access.
    '''
    basic_features = pd.read_csv('data/test_basic_features.csv')
    json_file = json.load(open('data/test.json', 'r'))
    uids = basic_features['uid']
    for uid in uids:
        csv = pd.read_csv('data/test_csvs/'+str(uid)+'.csv')
        dtypes = [np.int64, np.float64, np.float64]
        with open('data/test_npys/'+str(uid)+'.npy', 'wb+') as f:
            a = csv.to_numpy()
            np.save(f, a)

    basic_features = pd.read_csv('data/train_basic_features.csv')
    json_file = json.load(open('data/train.json', 'r'))
    uids = basic_features['uid']
    for uid in uids:
        csv = pd.read_csv('data/train_csvs/'+str(uid)+'.csv')
        dtypes = [np.int64, np.float64, np.float64]
        with open('data/train_npys/'+str(uid)+'.npy', 'wb+') as f:
            a = csv.to_numpy()
            np.save(f, a)
# data = get_uid_to_data('data/test_npys')
# print(data)
# df = data_to_df(data, LABELS)
# print(df)
