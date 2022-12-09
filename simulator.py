import itertools
import pandas as pd
import numpy as np
import os
import subprocess
import re
import time
import math
from alive_progress import alive_bar

from utils import *


def load_simulator(config):

    return Simulator(**config)


class Simulator:
    def __init__(self, ngspice_exec, test_netlist, parameter_list, performance_list, arguments, order,
                 sign):

        self.ngspice_exec = ngspice_exec
        self.test_netlist = os.path.join('config', 'circuits', test_netlist)
        self.arguments = dict(arguments)
        self.performance_list = list(performance_list)
        self.parameter_list = list(parameter_list)

        self.param_filenames = [str(x) + ".csv" for x in parameter_list]
        self.perform_filenames = [str(x) + ".csv" for x in performance_list]

        # validate arguments
        for p in parameter_list:

            assert (str(p) + "_start" in arguments.keys()), ("{} paramater must have a start index".format(p), arguments.keys())
            assert (str(p) + "_stop" in arguments.keys()), ("{} paramater must have a start index".format(p), arguments.keys())
            assert (str(p) + "_change" in arguments.keys()), (
                "{} paramater must have a start index".format(p), arguments.keys())

        self.save_error_log = False
        self.order = order
        self.sign = sign

    def runSimulation(self, parameters, train):
        assert type(parameters) is np.ndarray, "parameters should be np.array"
        assert parameters.shape[1] == len(self.parameter_list), f"list of points to simulate should be same length " \
                                                                f"as number of parameters {parameters.shape[1]} != " \
                                                                f"{len(self.parameter_list)} "
        num_params_to_sim = parameters.shape[0]
        MAX_SIM_SIZE = 500

        updated_netlist_filename = self.test_netlist.split("/")[-1] + "-formatted"
        tmp_out_path = os.path.join(os.getcwd(), "tmp_out")
        updated_netlist_filepath = os.path.join(tmp_out_path, updated_netlist_filename)
        argumentMap = self.arguments

        all_x, all_y = [], []

        if not train:
            argumentMap["out"] = tmp_out_path

        delete_testing_files(argumentMap["out"], [self.perform_filenames, self.param_filenames])
        size = math.ceil(num_params_to_sim / MAX_SIM_SIZE)
        
        with alive_bar(size) as bar:
            for i in range(size):  # sim in batches of MAX_SIM_SIZE (ngspice has a max input size)
                
                argumentMap["num_samples"] = parameters[i * MAX_SIM_SIZE:(i + 1) * MAX_SIM_SIZE, 0].shape[0]
                if argumentMap["num_samples"] == 0:
                    continue

                for param_index, p in enumerate(self.parameter_list):
                    argumentMap[f"{p}_array"] = " ".join(
                        list(parameters[i * MAX_SIM_SIZE:(i + 1) * MAX_SIM_SIZE, param_index].astype(str)))

                updateFile(self.test_netlist, updated_netlist_filepath, argumentMap)

                if self.save_error_log:
                    args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', "-o",
                            os.path.join(argumentMap["out"], "log.txt"), '-i',
                            updated_netlist_filepath]
                else:
                    args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', updated_netlist_filepath]

                with open(os.path.join(os.getcwd(), "tmp_out", 'out-file.txt'), 'w') as f:
                    subprocess.run(args,stdout=f,stderr=f)
                x, y = getData(self.param_filenames, self.perform_filenames, argumentMap["out"])

                all_x.append(x)
                all_y.append(y)
                bar()

            final_x = np.vstack(all_x)
            final_y = np.vstack(all_y)

            assert final_x.shape[
                    0] == num_params_to_sim, f"x has to few values. Original: {parameters.shape} X: {final_x.shape}"
            assert final_y.shape[
                    0] == num_params_to_sim, f"y has to few values. Original: {parameters.shape} Y: {final_y.shape}"

            if not train:
                delete_testing_files(argumentMap["out"], [self.perform_filenames, self.param_filenames])
            return final_x, final_y

    def load_data(self, train):
        if train:
            out_path = self.arguments["out"]
        else:
            out_path = os.path.join(os.getcwd(), "tmp_out")
        x, y = getData(self.param_filenames, self.perform_filenames, out_path)
        return x, y

