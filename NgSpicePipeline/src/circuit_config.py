from Simulator import *

def nmos_circuit(arguments_nmos = None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"

    train_netlist_nmos = "../assets/template/nmos/nmos-training.sp"
    test_netlist_nmos = "../assets/template/nmos/nmos-testing-pro.sp"
    param_list_nmos = ["nmos-r", "nmos-w"]
    perform_list_nmos = ["nmos-bw", "nmos-pw", "nmos-a0"]
    if order is None:
        order = [0,2,1]
    if sign is None:
        sign = [1,-1,1]
    if arguments_nmos is None:

        arguments_nmos = {
            "model_path": "../assets/model/45nm_CS.pm",
            "w_start": "2.88u",
            "w_stop": "6.63u",
            "w_change": "0.2u",
            "r_start": 620,
            "r_stop": 1450,
            "r_change": 5,
            "out": "../out/nmos-"
        }
    return Simulator(ngspice_exec, train_netlist_nmos, test_netlist_nmos, param_list_nmos, perform_list_nmos,
                               arguments_nmos, order, sign)


def cascode_circuit(arguments_cascode = None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_cascade = "../assets/template/cascode/nmos-training-cascode.sp"
    test_netlist_cascade = "../assets/template/cascode/nmos-testing-cascode.sp"
    param_list_cascade = ["cascode-r", "cascode-w0", "cascode-w1"]
    perform_list_cascade = ["cascode-bw", "cascode-pw", "cascode-a0"]
    if order is None:
        order = [0,2,1]
    if sign is None:
        sign = [1,-1,1]
    if arguments_cascode is None:
        arguments_cascode = {
            "model_path": "../assets/model/45nm_CS.pm",
            "w0_start": "4u",
            "w0_stop": "7.5u",
            "w0_change": "0.25u",
            "w1_start": "7u",
            "w1_stop": "10u",
            "w1_change": "0.2u",
            "r_start": 200,
            "r_stop": 500,
            "r_change": 18.75,
            "out": "../out/cascode-"
        }
    return Simulator(ngspice_exec, train_netlist_cascade, test_netlist_cascade, param_list_cascade,
                                  perform_list_cascade,
                                  arguments_cascode, order, sign)

def two_stage_circuit(arguments_two_stage = None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_two_stage = "../assets/template/two-stage/TwoStageAmplifier.sp"
    test_netlist_two_stage = "../assets/template/two-stage/TwoStageAmplifier-Test.sp"
    param_list_two_stage = ["ts-w0", "ts-w1", "ts-w2"]
    perform_list_two_stage = ["ts-bw", "ts-pw", "ts-a0"]
    if order is None:
        order = [0,2,1]
    if sign is None:
        sign = [1,-1,1]

    if arguments_two_stage is None:
        arguments_two_stage = {
            "model_path": "../assets/model/45nm_CS.pm",
            "w0_start": "25u",
            "w0_stop": "30u",
            "w0_change": "0.5u",
            "w2_start": "52u",
            "w2_stop": "55.5u",
            "w2_change": "0.5u",
            "w1_start": "6u",
            "w1_stop": "9u",
            "w1_change": "0.5u",
            "out": "../out/ts-"
        }
    simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
                                    perform_list_two_stage,
                                    arguments_two_stage, order, sign)
    simulator_two_stage.delete_existing_data = False
    return simulator_two_stage


def LNA_circuit(arguments_lna=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_lna = "../assets/template/LNA/LNA.sp"
    test_netlist_lna = "../assets/template/LNA/LNA_test.sp"
    param_list_lna = ["lna-ls", "lna-ld", "lna-lg", "lna-w"]
    perform_list_lna = ["lna-Gp", "lna-s11", "lna-nf"]

    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, -1, -1]

    if arguments_lna is None:
        arguments_lna = {
            "model_path": "../assets/model/45nm_CS.pm",
            "ls_start": "58.3p",
            "ls_stop": "60.3p",
            "ls_change": "0.25p",
            "ld_start": "4.4n",
            "ld_stop": "6n",
            "ld_change": "0.2n",
            "lg_start": "14.8n",
            "lg_stop": "16.4n",
            "lg_change": "0.2n",
            "w_start": "51u",
            "w_stop": "52.8u",
            "w_change": "0.3u",
            "out": "../out/lna-"
        }
    simulator_lna = Simulator(ngspice_exec, train_netlist_lna, test_netlist_lna, param_list_lna, perform_list_lna,
                              arguments_lna,order,sign)
    simulator_lna.delete_existing_data = False

    return simulator_lna

def VCO_circuit(arguments_vco=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_vco = "../assets/template/vco/VCOtraining.sp"
    test_netlist_vco = "../assets/template/vco/VCO.sp"
    param_list_vco = ["vco-w", "vco-w1", "vco-w2"]
    perform_list_vco = ["vco-power", "vco-pnoise", "vco-tuningrange"]

    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, -1, -1]

    if arguments_vco is None:
        arguments_vco = {
            "model_path": "../assets/model/45nm_CS.pm",
            "w_start": "4u",
            "w_stop": "5.4u",
            "w_change": "0.1u",
            "w1_start": "2u",
            "w1_stop": "10u",
            "w1_change": "0.5u",
            "w2_start": "15u",
            "w2_stop": "17.8u",
            "w2_change": "0.2u",
            "out": "../out/vco-"
        }
    simulator_vco = Simulator(ngspice_exec, train_netlist_vco, test_netlist_vco, param_list_vco, perform_list_vco,
                              arguments_vco,order,sign)
    simulator_vco.delete_existing_data = False

    return simulator_vco