import pandas as pd
import numpy as np
import os
import subprocess
import re


class Simulator:
    def __init__(self, ngspice_exec, train_netlist, test_netlist, parameter_list, performance_list, arguments):
        self.ngspice_exec = ngspice_exec
        self.train_netlist = train_netlist
        self.test_netlist = test_netlist
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
            file_data = file_data.apply(lambda x: re.split(r"\s+", str(x).replace("=",""))[2], axis=1)
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

        if self.delete_existing_data:
            self._delete_training_files()
            self._delete_testing_files()

        updated_netlist_filename = self.test_netlist + "-formatted"
        argumentMap = self.arguments
        argumentMap["num_samples"] = parameters.shape[0]
        for i, p in enumerate(self.parameter_list):
            argumentMap[f"{p}_array"] = " ".join(list(parameters[:, i].astype(str)))
        self._updateFile(self.test_netlist, updated_netlist_filename, argumentMap)
        if self.save_error_log:
            args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', "-o", os.path.join(self.arguments["out"]), '-i',
                    updated_netlist_filename]
        else:
            args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', updated_netlist_filename]
        subprocess.run(args)
        x, y = self.getData(self.test_param_filenames, self.test_perform_filenames, argumentMap["out"])
        return [x, y]

    def run_training(self):
        if self.delete_existing_data:
            self._delete_training_files()
            self._delete_testing_files()

        formatted_netlist = self.train_netlist + "-formatted"
        self._updateFile(self.train_netlist, formatted_netlist, self.arguments)
        if self.save_error_log:
            args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', "-o",
                    os.path.join(self.arguments["out"], "log.txt"), formatted_netlist]
        else:
            args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', formatted_netlist]
        subprocess.run(args)
        x, y = self._getData(self.train_param_filenames, self.train_perform_filenames, self.arguments["out"])

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

    def _delete_testing_files(self):
        out = self.arguments["out"]
        param_fullname = [os.path.join(out, file) for file in self.test_param_filenames]
        perform_fullname = [os.path.join(out, file) for file in self.test_perform_filenames]

        for file in (param_fullname + perform_fullname):
            try:
                os.remove(file)
            except FileNotFoundError:
                continue


if __name__ == '__main__':
    # TODO: remove print statements
    # TODO: find work around for too many arguments crashing simulator
    # TODO: find way to include simulator console logging for easier debugging (try -o output filename)
    # TODO: connect with training
    ngspice_exec = "ngspice/Spice64/bin/ngspice.exe"
    train_netlist = "NgSpicePipeline/assets/nmos-training-2.sp"
    test_netlist = "NgSpicePipeline/assets/nmos-testing-pro.sp"
    param_list = ["r", "w"]
    perform_list = ["bw", "pw", "a0"]

    arguments = {
        "model_path": "NgSpicePipeline/assets/45nm_CS.pm",
        "start1": "2.88u",
        "stop1": "6.63u",
        "change1": "0.3750u",
        "start2": 620,
        "stop2": 1450,
        "change2": 5.5,
        "out": "NgSpicePipeline/out/"
    }

    simulator_nmos = Simulator(ngspice_exec, train_netlist, test_netlist, param_list, perform_list, arguments)
    simulator = simulator_nmos
    rerun_training = True
    if rerun_training:
        x, y = simulator.run_training()
    else:
        param_outfile_names = ["r.csv", "w.csv"]  # must be in order
        perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order
        out = "../out/"
        x, y = simulator.getData(param_outfile_names, perform_outfile_names, out)

    data = np.hstack((x, y)).astype(float)

    x1, x2 = x[:, 0], x[:, 1]

    x_sim, y_sim = simulator.runSimulation(x)
    print(x.shape, x_sim.shape)
    print("x[0], y[0]", x[0], y[0])
    print("x_sim[0], y_sim[0]", x_sim[0], y_sim[0])

    for i in range(x.shape[0]):
        assert np.all(x[i] == x_sim[i]), (x[i], x_sim[i])
        assert np.all(y[i] == y_sim[i]), (y[i], y_sim[i])
