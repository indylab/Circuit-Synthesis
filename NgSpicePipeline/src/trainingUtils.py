import pandas as pd
import numpy as np
import os
import subprocess
import re


def updateFile(trainingfilePath, outputFilePath, argumentMap):
    with open(trainingfilePath, 'r') as read_file:
        file_content = read_file.read()
        for key, val in argumentMap.items():
            temp_pattern = "{" + str(key) + "}"
            file_content = file_content.replace(temp_pattern, str(val))

        with open(outputFilePath, 'w') as write_file:
            write_file.write(file_content)


def convert(filenames):
    files = []
    for file in filenames:
        file_data = pd.read_csv(file, header=None)
        files.append(file_data.apply(lambda x: re.split(r"\s+", str(x))[2], axis=1))

    combine = pd.concat(files, axis=1)
    return np.array(combine, dtype=float)


def getData(param_outfile_names, perform_outfile_names, out):
    param_fullname = [os.path.join(out, file) for file in param_outfile_names]
    perform_fullname = [os.path.join(out, file) for file in perform_outfile_names]
    x = convert(param_fullname)
    y = convert(perform_fullname)
    return x, y


def runSimulation(x1_list, x2_list):
    netlist = "NgSpicePipeline/assets/nmos-testing-pro.sp"
    updated_netlist = "NgSpicePipeline/assets/formatted-nmos-testing.sp"
    pm = "NgSpicePipeline/assets/45nm_CS.pm"
    argumentMap = {
        "model_path": pm,
        "r_array": " ".join(list(x1_list.astype(str))),
        "w_array": " ".join(list(x2_list.astype(str))),
        "num_samples": len(x2_list),
        "out": "NgSpicePipeline/out"
    }
    updateFile(netlist, updated_netlist, argumentMap)

    ngspice_exec = "ngspice/Spice64/bin/ngspice.exe"
    subprocess.run([ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', updated_netlist])

    param_outfile_names = ["r-test.csv", "w-test.csv"]  # must be in order
    perform_outfile_names = ["bw-test.csv", "pw-test.csv", "a0-test.csv"]  # must be in order

    x, y = getData(param_outfile_names, perform_outfile_names, argumentMap["out"])
    return [x, y]


def run_training():
    arguments = {
        "model_path": "NgSpicePipeline/assets/45nm_CS.pm",
        "start1": 620,
        "stop1": 1450,
        "change1": 11,
        "start2": "2.88u",
        "stop2": "6.63u",
        "change2": "0.3750u",
        "out": "NgSpicePipeline/out"
    }
    netlist = "NgSpicePipeline/assets/nmos-training.sp"
    formatted_netlist = "NgSpicePipeline/assets/nmos-training.sp"
    updateFile(netlist, formatted_netlist, arguments)

    ngspice_exec = "ngspice/Spice64/bin/ngspice.exe"
    subprocess.run([ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', formatted_netlist])

    param_outfile_names = ["r.csv", "w.csv"]  # must be in order
    perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order

    x, y = getData(param_outfile_names, perform_outfile_names, arguments["out"])

    return x, y


if __name__ == '__main__':
    rerun_training = False
    if rerun_training:
        x, y = run_training()
    else:
        param_outfile_names = ["r.csv", "w.csv"]  # must be in order
        perform_outfile_names = ["bw.csv", "pw.csv", "a0.csv"]  # must be in order
        out = "NgSpicePipeline/out"
        x, y = getData(param_outfile_names, perform_outfile_names, out)

    data = np.hstack((x, y)).astype(float)

    x1, x2 = x[:, 0], x[:, 1]

    x_sim, y_sim = runSimulation(x1, x2)
    print(x.shape, x_sim.shape)
    print(x[0], y[0])
    print(x_sim[0], y_sim[0])

    for i in range(x.shape[0]):
        assert np.all(x[i] == x_sim[i]),  (x[i] == x_sim[i])
        assert np.all(y[i] == y_sim[i]), (y[i] == y_sim[i])
