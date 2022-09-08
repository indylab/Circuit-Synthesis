from Simulator import *

def nmos_circuit(arguments_nmos = None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"

    train_netlist_nmos = "../assets/nmos-training.sp"
    test_netlist_nmos = "../assets/nmos-testing-pro.sp"
    param_list_nmos = ["r", "w"]
    perform_list_nmos = ["bw", "pw", "a0"]
    if order is None:
        order = [0,2,1]
    if sign is None:
        sign = [1,-1,1]
    if arguments_nmos is None:
        arguments_nmos = {
            "model_path": "../assets/45nm_CS.pm",
            "w_start": 620,
            "w_stop": 1450,
            "w_change": 5,
            "r_start": "2.88u",
            "r_stop": "6.63u",
            "r_change": "0.20u",
            "out": "../out/"
        }
    return Simulator(ngspice_exec, train_netlist_nmos, test_netlist_nmos, param_list_nmos, perform_list_nmos,
                               arguments_nmos, order, sign)


def cascade_circuit(arguments_cascade = None, order=None, sign=None):
    train_netlist_cascade = "../assets/nmos-training-cascode.sp"
    test_netlist_cascade = "../assets/nmos-testing-cascode.sp"
    param_list_cascade = ["r", "w0", "w1"]
    perform_list_cascade = ["bw", "pw", "a0"]
    if order is None:
        order = [0,2,1]
    if sign is None:
        sign = [1,-1,1]
    if arguments_cascade is None:
        arguments_cascade = {
            "model_path": "../assets/45nm_CS.pm",
            "w0_start": 620,
            "w0_stop": 1450,
            "w0_change": 50,
            "w1_start": 620,
            "w1_stop": 1450,
            "w1_change": 50,
            "r_start": "2.88u",
            "r_stop": "6.63u",
            "r_change": "0.7500u",
            "out": "../out/"
        }
    return Simulator(ngspice_exec, train_netlist_cascade, test_netlist_cascade, param_list_cascade,
                                  perform_list_cascade,
                                  arguments_cascade, order, sign)

def two_stage_circuit(arguments_two_stage = None, order=None, sign=None):
    train_netlist_two_stage = "../assets/TwoStageAmplifier.sp"
    test_netlist_two_stage = "../assets/TwoStageAmplifier-Test.sp"
    param_list_two_stage = ["w0", "w1", "w2"]
    perform_list_two_stage = ["bw", "pw", "a0"]
    if order is None:
        order = [0,2,1]
    if sign is None:
        sign = [1,-1,1]

    if arguments_two_stage is None:
        arguments_two_stage = {
            "model_path": "../assets/45nm_CS.pm",
            "w0_start": "25u",
            "w0_stop": "30u",
            "w0_change": "0.5u",
            "w2_start": "52u",
            "w2_stop": "55.5u",
            "w2_change": "0.5u",
            "w1_start": "6u",
            "w1_stop": "9u",
            "w1_change": "0.5u",
            "out": "../out/"
        }
    simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
                                    perform_list_two_stage,
                                    arguments_two_stage, order, sign)
    simulator_two_stage.delete_existing_data = True
    return simulator_two_stage


def LNA_circuit(arguments_lna=None, order=None, sign=None):
    train_netlist_lna = "../assets/LNA.sp"
    test_netlist_lna = "../assets/LNA_test"
    param_list_lna = ["ls", "ld", "lg", "r", "w"]
    perform_list_lna = ["Gmax", "Gp", "s11", "nf"]
    if order is None:
        order = [0,1,2,3]
    if sign is None:
        sign = [1,1,1,1]
    if arguments_lna is None:
        arguments_lna = {
            "model_path": "../assets/45nm_CS.pm",
            "ls_start": "58.3p",
            "ls_stop": "60.8p",
            "ls_change": "0.5p",
            "ld_start": "4.4n",
            "ld_stop": "5.4n",
            "ld_change": "0.5n",
            "lg_start": "14.8n",
            "lg_stop": "15.8n",
            "lg_change": "0.24n",
            "r_start": "800",
            "r_stop": "1050",
            "r_change": "50",
            "w_start": "51u",
            "w_stop": "53u",
            "w_change": "0.4u",
            "out": "../out/"
        }
    return Simulator(ngspice_exec, train_netlist_lna, test_netlist_lna, param_list_lna, perform_list_lna,
                              arguments_lna, order, sign)