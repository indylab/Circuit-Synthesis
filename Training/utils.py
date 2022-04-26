import os
import pandas as pd
import numpy as np
from torch.utils.data import random_split
import pickle
from sklearn import preprocessing
DEFAULT_FILE_PATH = '../Data/BW_Gain.csv'


def parseGainAndBWCsv(srcFile: str, discard = True) -> list:
    if os.path.exists(srcFile):
        dt = pd.read_csv(srcFile)
        width_column = dt['width']
        # store each row transistor width value
        row_index_dict = dict()

        # store each column resistor load value
        column_index_dict = dict()

        for index, row in enumerate(width_column):
            row_index_dict[index] = row

        for index, column in enumerate(
                dt.columns[1:len(dt.columns):2]):  # skip every other column cause of joint csv table
            column_index_dict[index] = column

        value_dt = dt.loc[:, dt.columns != 'width']
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

def parseGainAndBWCsv2(srcFile):
    try: 
        os.path.exists(srcFile)
    except FileNotFoundError:
        print(f"FIle {srcFile} doesnt exist")
        raise FileNotFoundError
    
    data = pd.io.parsers.read_csv(srcFile)
    resistors = data.keys()[::2][1:] # get column labels (every other col minus 1st)
    widths = data.iloc[:,0] # get width values from first column

    values = data.values[:,1:] # get values minus first column (row labels)

    res = []
    for r in range(len(resistors)):
        for w in range(len(widths)):
            p = [resistors[r], widths[w], values[w,r*2], values[w,r*2+1]]
            if " " in p:  # ignore values with null values
                continue
            res.append(p)
    res = np.array(res)
    return res
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
    assert data.shape[1] == 4, "reshape the data first to (-1, 4)"
    normed_data = np.zeros(data.shape)
    normers = []
    for i in range(len(data.T)):
        normer = preprocessing.MinMaxScaler((-1,1))
        normed_data.T[i] = normer.fit_transform(data.T[i].reshape(-1,1)).reshape(-1)
        normers.append(normer)
    return normed_data,np.array(normers)

def denormalize(normed_data,normers):
    denormed_data = np.zeros(normed_data.shape)
    for i in range(len(normed_data.T)):
        denormed_data.T[i] = normers[i].inverse_transform(normed_data.T[i].reshape(-1,1)).reshape(-1)
    return denormed_data

def dropCollisions(data, perform_sim, param_sim):
    assert(data.shape[1] == 2), "data should be of shape ([x1],[y1]) ... ([xn],[yn])"
    e = 0.01
    alias = 0
    diffs = []
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            x1 = data[i][0] # params 1
            y1 = data[i][1] # performance 1
            x2 = data[j][0] # params 2
            y2 = data[j][1] # performance 2
            # print("*"*100)
            # print(data[i])
            # print(data[j])
            diff_y = np.linalg.norm(y1 - y2) # 
            diff_x = np.linalg.norm(x1 - x2)
            
            diffs.append([diff_y,diff_x,(i,j)])
    diffs = np.array(diffs,dtype=object)
    print("Dropping Collisions")
    print("Size of Data", data.shape)
    print("Size of Diffs", len(diffs))
    
    to_del = set([])
    for pair in diffs:
        if pair[0] > perform_sim and pair[1] < param_sim:
            to_del = to_del.union(set(pair[2]))
            
    for i in sorted(list(to_del))[::-1]:
        data = np.delete(data,i,0)
    print("dropped",(len(to_del)))
    return data
            
    
    
if __name__ == '__main__':
    k = parseGainAndBWCsv(DEFAULT_FILE_PATH)
    print(k)