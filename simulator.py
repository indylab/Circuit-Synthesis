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
        self.arguments = arguments
        self.performance_list = performance_list
        self.parameter_list = parameter_list

        self.param_filenames = [str(x) + ".csv" for x in parameter_list]
        self.perform_filenames = [str(x) + ".csv" for x in performance_list]

        # validate arguments
        for p in parameter_list:
            assert (str(p) + "_start" in arguments.keys()), ("Each paramater must have a start index", arguments.keys())
            assert (str(p) + "_stop" in arguments.keys()), ("Each paramater must have a start index", arguments.keys())
            assert (str(p) + "_change" in arguments.keys()), (
                "Each paramater must have a start index", arguments.keys())

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

                with open('out-file.txt', 'w') as f:
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




class MultiSimulator:
    def __init__(self, ngspice_exec, test_netlists, parameter_list, performance_list, arguments, order,
                 sign):
        self.ngspice_exec = ngspice_exec

        self.test_netlist = test_netlists
        self.arguments = arguments
        self.performance_list = performance_list
        self.parameter_list = parameter_list

        # create output filenames
        self.train_param_filenames = [str(x) + ".csv" for x in parameter_list]
        self.train_perform_filenames = [str(x) + ".csv" for x in performance_list]

        self.test_param_filenames = [str(x) + "-test.csv" for x in parameter_list]
        self.test_perform_filenames = [str(x) + "-test.csv" for x in performance_list]

        self.delete_existing_data = False
        # validate arguments
        for p in parameter_list:
            assert (str(p) + "_start" in arguments.keys()), ("Each paramater must have a start index", arguments.keys())
            assert (str(p) + "_stop" in arguments.keys()), ("Each paramater must have a start index", arguments.keys())
            assert (str(p) + "_change" in arguments.keys()), (
                "Each paramater must have a start index", arguments.keys())

        self.save_error_log = False
        self.order = order
        self.sign = sign

    def _updateFile(self, trainingFilePath, outputFilePath, argumentMap):
        with open(trainingFilePath, 'r') as read_file:
            file_content = read_file.read()
            for key, val in argumentMap.items():
                temp_pattern = "{" + str(key) + "}"
                file_content = file_content.replace(temp_pattern, str(val))

            with open(outputFilePath, 'w') as write_file:
                write_file.write(file_content)

    @staticmethod
    def _convert(filenames):
        files = []
        for file in filenames:
            file_data = pd.read_csv(file, header=None)
            file_data = file_data.apply(lambda x: re.split(r"\s+", str(x).replace("=", ""))[2], axis=1)
            files.append(file_data)
        combine = pd.concat(files, axis=1)
        return np.array(combine, dtype=float)

    def getData(self, param_outfile_names, perform_outfile_names, out):
        param_fullname = [os.path.join(out, file) for file in param_outfile_names]
        perform_fullname = [os.path.join(out, file) for file in perform_outfile_names]
        x = self._convert(param_fullname)
        y = self._convert(perform_fullname)
        return x, y

    def _getData(self, param_outfile_names, perform_outfile_names, out):
        param_fullname = [os.path.join(out, file) for file in param_outfile_names]
        perform_fullname = [os.path.join(out, file) for file in perform_outfile_names]
        x = self._convert(param_fullname)
        y = self._convert(perform_fullname)
        return x, y

    def runSimulation(self, parameters):
        assert type(parameters) is np.ndarray, "parameters should be np.array"
        assert parameters.shape[1] == len(self.parameter_list), f"list of points to simulate should be same length " \
                                                                f"as number of parameters {parameters.shape[1]} != " \
                                                                f"{len(self.parameter_list)} "
        num_params_to_sim = parameters.shape[0]
        MAX_SIM_SIZE = 750

        if self.delete_existing_data:
            self._delete_training_files()
            self._delete_testing_files()

        argumentMap = self.arguments
        all_x, all_y = [], []

        for j in range(math.ceil(
                num_params_to_sim / MAX_SIM_SIZE)):  # sim in batches of MAX_SIM_SIZE (ngspice has a max input size)
            argumentMap["num_samples"] = parameters[j * MAX_SIM_SIZE:(j + 1) * MAX_SIM_SIZE, 0].shape[0]

            if argumentMap["num_samples"] == 0:
                continue
            for param_index, p in enumerate(self.parameter_list):
                argumentMap[f"{p}_array"] = " ".join(
                    list(parameters[j * MAX_SIM_SIZE:(j + 1) * MAX_SIM_SIZE, param_index].astype(str)))
            for i in range(len(self.test_netlist)):
                updated_netlist_filename = self.test_netlist[i] + "-formatted"
                print(updated_netlist_filename)
                self._updateFile(self.test_netlist[i], updated_netlist_filename, argumentMap)
                if self.save_error_log:
                    args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', "-o",
                            os.path.join(self.arguments["out"], "log.txt"), '-i',
                            updated_netlist_filename]
                else:
                    args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', updated_netlist_filename]
                subprocess.run(args)
                if i != len(self.test_netlist)-1:  # last index
                    print(i)
                    self._delete_param_testing_files()
                print(i)
            x, y = self.getData(self.test_param_filenames, self.test_perform_filenames, argumentMap["out"])
            self._delete_testing_files()
            self._delete_training_files()
            print(x.shape,y.shape)
            all_x.append(x)
            all_y.append(y)

        final_x = np.vstack(all_x)
        final_y = np.vstack(all_y)
        assert final_x.shape[
                   0] == num_params_to_sim, f"x has to few values. Original: {parameters.shape} X: {final_x.shape}"
        assert final_y.shape[
                   0] == num_params_to_sim, f"y has to few values. Original: {parameters.shape} Y: {final_y.shape}"

        return [final_x, final_y]


    def run_training(self):
        if self.delete_existing_data:
            self._delete_training_files()
            self._delete_testing_files()

        all_ranges = []

        value_reg = r"[0-9]+\.?[0-9]*"
        unit_reg = r"[a-z][A-Z]*"

        for param in self.parameter_list:
            start_raw = self.arguments[f"{param}_start"]
            start = float(re.findall(value_reg, start_raw)[0])
            regex_list = re.findall(unit_reg, start_raw)
            start_unit = regex_list[0] if len(regex_list) != 0 else ''

            stop_raw = self.arguments[f"{param}_stop"]
            stop = float(re.findall(value_reg, stop_raw)[0])
            regex_list = re.findall(unit_reg, start_raw)
            stop_unit = regex_list[0] if len(regex_list) != 0 else ''

            change_raw = self.arguments[f"{param}_change"]
            change = float(re.findall(value_reg, change_raw)[0])
            regex_list = re.findall(unit_reg, start_raw)
            change_unit = regex_list[0] if len(regex_list) != 0 else ''

            assert (start_unit == stop_unit == change_unit), f"not the same for all parts of range: parameter: " \
                                                             f"{param}, start {stop_unit}, stop {stop_unit}, " \
                                                             f"change {change_unit} "

            param_range = []
            curr = start
            while curr <= stop:
                param_range.append(str(curr) + stop_unit)
                curr += change

            all_ranges.append(list(param_range))

        train_data = np.array(list(itertools.product(*all_ranges)))
        print(f"training data size = {train_data.shape}")

        x, y = self.runSimulation(train_data)

        return x, y

    def _delete_training_files(self):
        out = self.arguments["out"]
        param_fullname = [os.path.join(out, file) for file in self.train_param_filenames]
        perform_fullname = [os.path.join(out, file) for file in self.train_perform_filenames]

        for file in (param_fullname + perform_fullname):
            try:
                os.remove(file)
            except FileNotFoundError:
                continue

    def _delete_param_testing_files(self):
        out = self.arguments["out"]
        param_fullname = [os.path.join(out, file) for file in self.test_param_filenames]

        for file in (param_fullname):
            try:
                os.remove(file)
            except FileNotFoundError:
                continue
            except PermissionError:
                time.sleep(1)
                os.remove(file)

    def _delete_testing_files(self):
        out = self.arguments["out"]
        param_fullname = [os.path.join(out, file) for file in self.test_param_filenames]
        perform_fullname = [os.path.join(out, file) for file in self.test_perform_filenames]

        for file in (param_fullname + perform_fullname):
            try:
                os.remove(file)
            except FileNotFoundError:
                continue
            except PermissionError:
                time.sleep(1)
                os.remove(file)
