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


def convert(filename1, filename2, filename3):
    file1 = pd.read_csv(filename1, header=None)
    file2 = pd.read_csv(filename2, header=None)
    file3 = pd.read_csv(filename3, header=None)

    file1 = file1.apply(lambda x: str(x).split()[2], axis=1)
    file2 = file2.apply(lambda x: str(x).split()[2], axis=1)
    file3 = file3.apply(lambda x: str(x).split()[2], axis=1)

    print(file1.shape)
    combine = pd.concat([file1, file2, file3], axis=1)
    print(np.array(combine))

    return np.array(combine,dtype=float)


def getX(start1, stop1, change1, start2, stop2, change2):
    curr1 = start1
    curr2 = start2

    X = []
    while curr1 < stop1:
        while curr2 <= stop2:
            X.append([curr1, curr2])
            curr2 += change2
        curr2 = start2
        curr1 += change1

    X = np.array(X, dtype=float)

    return X


def runSimulation(x1, x2):
    netlist = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\assets\\nmos-testing.sp"
    updated_netlist = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\assets\\formatted-nmos-testing.sp"
    pm = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\assets\\45nm_CS.pm"
    argumentMap = {
        "model_path": pm,
        "var1": str(x1),
        "var2": str(x2) + "u",
        "out": "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\out\\"
    }
    updateFile(netlist, updated_netlist, argumentMap)

    ngspice_exec = "C:\\Users\\tjtom\\Downloads\\ngspice-28_64\\Spice64\\bin\\ngspice.exe"
    subprocess.run([ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', updated_netlist])

    outfile_names = ["bw-test.csv", "pw-test.csv", "a0-test.csv"]  # must be in order

    data_points = []
    for file in outfile_names:
        full_filename = os.path.join(argumentMap["out"], file)
        text = open(full_filename).read()
        text = re.split(r"\s+", text)
        print(text)
        data_points.append(text[2])
    return np.array(data_points,dtype=float)


def run_training():
    arguments = {
        "model_path": "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\assets\\45nm_CS.pm",
        "start1": 620,
        "stop1": 1450,
        "change1": 11,
        "start2": "2.88u",
        "stop2": "6.63u",
        "change2": "0.3750u",
        "out": "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\out\\"
    }
    netlist = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\assets\\nmos-training.sp"
    formatted_netlist = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\assets\\formatted-nmos-training.sp"
    updateFile(netlist, formatted_netlist, arguments)

    ngspice_exec = "C:\\Users\\tjtom\\Downloads\\ngspice-37_64\\Spice64\\bin\\ngspice.exe"
    subprocess.run([ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', formatted_netlist])


if __name__ == '__main__':
    #run_training()
    f1 = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\out\\bw.csv"
    f2 = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\out\\pw.csv"
    f3 = "C:\\Users\\tjtom\\OneDrive\\Desktop\\File_Cabinet\\Code\\SimulationStuff\\out\\a0.csv"
    y = convert(f1, f2, f3)

    inp = [620, 1450, 11, 2.88, 6.63, 0.3750]

    x = getX(*inp)
    print("x.shape:", x.shape)

    data = np.hstack((x, y)).astype(float)

    test_point = x[0]
    print(test_point)
    y_sim = runSimulation(*test_point)

    print("true y:", y[0])
    print("sim y:", y_sim)

    for i in range(y.shape[0]):
        if np.allclose(y[i],y_sim):
            print("y[i]",y[i])
            print("y_sim",y_sim)

