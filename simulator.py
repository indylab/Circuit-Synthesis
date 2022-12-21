import itertools
import pandas as pd
import numpy as np
import os
import subprocess
import re
import time
import math
from alive_progress import alive_bar

from multiprocessing import Pool
from functools import partial
from utils import *


def load_simulator(circuit_config,simulator_config):

    return Simulator(simulator_config,**circuit_config)


class Simulator:
    def __init__(self, simulator_config, ngspice_exec, test_netlist, parameter_list, performance_list, arguments, order,
                 sign,):

       
        self.ngspice_exec = ngspice_exec
        self.test_netlist = os.path.join('config', 'circuits', test_netlist)
        self.arguments = dict(arguments)
        self.performance_list = list(performance_list)
        self.parameter_list = list(parameter_list)

        self.num_params = len(parameter_list)
        self.num_perf = len(performance_list)

        self.param_filenames = [str(x) + ".csv" for x in parameter_list]
        self.perform_filenames = [str(x) + ".csv" for x in performance_list]
        self.MAX_SIM_SIZE = simulator_config['sim_size']
        self.num_workers = simulator_config['num_workers']
        self.multiprocessing = simulator_config['multiprocessing'] 
        print(f'Number of Workers, {self.num_workers}')
        print(f'MAX_SIM_SIZE, {self.MAX_SIM_SIZE}')

        # validate arguments
        for p in parameter_list:

            assert (str(p) + "_start" in arguments.keys()), ("{} paramater must have a start index".format(p), arguments.keys())
            assert (str(p) + "_stop" in arguments.keys()), ("{} paramater must have a start index".format(p), arguments.keys())
            assert (str(p) + "_change" in arguments.keys()), (
                "{} paramater must have a start index".format(p), arguments.keys())

        self.save_error_log = False
        self.order = order
        self.sign = sign

    def process_batch(self,parameters,argumentMap,updated_netlist_filepath, batch_index):
        #ArgumentMap is also forked
        # print(argumentMap['out'])
        # argumentMap['out'] = os.path.join(argumentMap['out'],f'batch_{batch_index}')
        
        argumentMap["num_samples"] = parameters[batch_index * self.MAX_SIM_SIZE:(batch_index + 1) * self.MAX_SIM_SIZE, 0].shape[0]
        argumentMap['batch_index'] = batch_index
        
        if argumentMap["num_samples"] == 0:
            return

        path = os.path.join(argumentMap['out'],f'batch_{batch_index}')

        if not os.path.exists(path):
            os.mkdir(path)


        for param_index, p in enumerate(self.parameter_list):
            argumentMap[f"{p}_array"] = " ".join(
                list(parameters[batch_index * self.MAX_SIM_SIZE:(batch_index + 1) * self.MAX_SIM_SIZE, param_index].astype(str)))
            
        file_name = updateFile(self.test_netlist, updated_netlist_filepath, argumentMap,batch_index,path)
        args = [self.ngspice_exec, '-r', 'rawfile.raw', '-b', '-i', file_name]

        with open(os.path.join(os.getcwd(), "tmp_out", f'out-file_{batch_index}.txt'), 'w') as f:
            subprocess.run(args,stdout=f,stderr=f)
        
        final_x,final_y = getData(self.param_filenames, self.perform_filenames, path)
        
        return final_x,final_y

    def runSimulation(self, parameters, train):
        assert type(parameters) is np.ndarray, "parameters should be np.array"
        assert parameters.shape[1] == len(self.parameter_list), f"list of points to simulate should be same length " \
                                                                f"as number of parameters {parameters.shape[1]} != " \
                                                                f"{len(self.parameter_list)} "
        num_params_to_sim = parameters.shape[0]
        

        updated_netlist_filename = self.test_netlist.split("/")[-1] + "-formatted"
        tmp_out_path = os.path.join(os.getcwd(), "tmp_out")
        updated_netlist_filepath = os.path.join(tmp_out_path, updated_netlist_filename)
        argumentMap = self.arguments
        
        if not train:
            argumentMap["out"] = tmp_out_path
            
        delete_testing_files(argumentMap["out"], [self.perform_filenames, self.param_filenames])
        size = math.ceil(num_params_to_sim / self.MAX_SIM_SIZE)
        start = time.time()
        # pbar = alive_bar(total=len(tasks))


        if self.multiprocessing:
            print(f'Starting MP with simulation size of {num_params_to_sim} and {size} batches')

            with Pool(processes=self.num_workers) as pool:
                #Some functional magick
                process_partial = partial(self.process_batch,parameters,argumentMap,updated_netlist_filepath)
                #Run the simulation
                out_data = pool.map(process_partial, range(size))
        else:
            print("Running without multiprocessing")
            out_data = []
            for i in range(size):
                out_data.append(self.process_batch(parameters,argumentMap,updated_netlist_filepath,i))
        
        final_x = []
        final_y = []
        for x,y in out_data:
            if x is not None:
                final_x.append(x)
                final_y.append(y)
        
        final_x = np.concatenate(final_x)
        final_y = np.concatenate(final_y)

        print('MP took', time.time() - start, 'seconds')
        

        # final_x, final_y = getData(self.param_filenames, self.perform_filenames, argumentMap["out"])
        assert final_x.shape[
                0] == num_params_to_sim, f"x has to few values. Original: {parameters.shape} X: {final_x.shape}"
        assert final_y.shape[
                0] == num_params_to_sim, f"y has to few values. Original: {parameters.shape} Y: {final_y.shape}"
        
        if train:
            #Save the data
            print("Saving Data to ", argumentMap["out"])
            np.save(os.path.join(argumentMap["out"], "x.npy"), final_x)
            np.save(os.path.join(argumentMap["out"], "y.npy"), final_y)

        delete_testing_files(argumentMap["out"], [self.perform_filenames, self.param_filenames])
        return final_x, final_y

    def load_data(self, train):
        if train:
            out_path = self.arguments["out"]
        else:
            out_path = os.path.join(os.getcwd(), "tmp_out")
        x, y = getData(self.param_filenames, self.perform_filenames, out_path)
        return x, y

