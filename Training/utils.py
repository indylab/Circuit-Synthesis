import os
import pandas as pd
import numpy as np
from torch.utils.data import random_split
import pickle
DEFAULT_FILE_PATH = '../Data/BW_Gain.csv'


def parseGainAndBWCsv(srcFile: str, discard = True) -> list:
    if os.path.exists(srcFile):
        dt = pd.read_csv(srcFile)
        width_column = dt['Width']
        # store each row transistor width value
        row_index_dict = dict()

        # store each column resistor load value
        column_index_dict = dict()

        for index, row in enumerate(width_column):
            row_index_dict[index] = row

        for index, column in enumerate(
                dt.columns[1:len(dt.columns):2]):  # skip every other column cause of joint csv table
            column_index_dict[index] = column

        value_dt = dt.loc[:, dt.columns != 'Width']
        data_list = []

        for row in range(len(dt)):
            for column in range(0, len(value_dt.columns), 2):
                bw_raw = value_dt.iloc[row, column]
                gain_raw = value_dt.iloc[row, column + 1]

                # if nan value is -1
                bandwidth = bw_raw if not pd.isnull(bw_raw) else -1.0
                gain = gain_raw if not pd.isnull(gain_raw) else -1.0
                
                if bandwidth == -1.0 or gain == -1.0:
                    continue
                
                try:
                    data_tuple = ([float(row_index_dict[row]), float(column_index_dict[column // 2])], [float(bandwidth), float(gain)])
                    data_list.append(data_tuple)
                except:
                    continue
        # Each data point will be a tuple with a format of (Transistor width, resistor load, Bandwidth, Gain)
        return data_list
    else:
        raise FileNotFoundError


def mockSimulator(xy):
    np.random.seed(123)
    input = xy

    A = np.random.rand(1, 5)
    B = np.random.rand(5, 2)
    C = np.random.rand(2, 15)

    iA = input.dot(A)
    iAB = iA.dot(B)
    full = iAB.dot(C)

    ret = np.array([np.mean(full[0]), np.mean(full[1])]).reshape(2, 1)

    return ret


def splitDataset(dataset, train_size_prob):
    train_size = int(len(dataset) * train_size_prob)

    train_dataset, validate_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    return train_dataset, validate_dataset

def openPickle(filename):
    infile = open(filename,'rb')
    infile.seek(0)
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    norm_data = (data-data_min)/(data_max-data_min)
    return norm_data, data_min, data_max

def denormalize(data,data_min,data_max):
    denorm_data = (data * (data_max - data_min) + data_min)
    return denorm_data
    
if __name__ == '__main__':
    k = parseGainAndBWCsv(DEFAULT_FILE_PATH)
    print(k)
