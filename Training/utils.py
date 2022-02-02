import os
import pandas as pd
import numpy as np

DEFAULT_FILE_PATH = '../Data/BW_Gain.csv'


def parseGainAndBWCsv(srcFile: str) -> list:
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

                try:
                    data_tuple = ([row_index_dict[row], float(column_index_dict[column // 2])], [bandwidth, gain])
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
