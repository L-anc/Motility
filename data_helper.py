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

def get_uid_to_data(file_path):
    uids = [g.removeprefix(file_path + "/").removesuffix(".npy") for g in glob(f'{file_path}/*.npy')]
    return { u:np.load(file_path + "/" + u + ".npy") for u in uids}


