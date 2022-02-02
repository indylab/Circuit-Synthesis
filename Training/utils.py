import os
import pandas as pd
import numpy as np

DEFAULT_FILE_PATH = '../Data/BW_Gain.csv'

def parseGainAndBWCsv(srcFile: str) -> list:
    if os.path.exists(srcFile):
        dt = pd.read_csv(srcFile)
        width_column = dt['Width']

        #store each row transistor width value
        row_index_dict = dict()

        #store each column resistor load value
        column_index_dict = dict()

        for index, row in enumerate(width_column):
            row_index_dict[index] = row

        for index, column in enumerate(dt.columns[1:len(dt.columns):2]):
            column_index_dict[index] = column

        value_dt = dt.loc[:, dt.columns != 'Width']

        data_list = []

        for row in range(len(dt)):
            for column in range(0, len(value_dt.columns),2):
                if not pd.isnull(value_dt.iloc[row, column]) and not pd.isnull(value_dt.iloc[row, column + 1]):
                    try:
                        data_tuple = (row_index_dict[row], float(column_index_dict[column//2]),
                                      value_dt.iloc[row, column], value_dt.iloc[row, column + 1])
                        data_list.append(data_tuple)
                    except:
                        continue

        # Each data point will be a tuple with a format of (Transistor width, resistor load, Bandwidth, Gain)
        return data_list
    else:
        raise FileNotFoundError























if __name__ == '__main__':
    k = parseGainAndBWCsv(DEFAULT_FILE_PATH)
    print(k)