from NgSpicePipeline.src.MultiSimulator import MultiSimulator
from Simulator import *
import os

out_path = os.path.abspath("../out")


def nmos_circuit(arguments_nmos=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"

    train_netlist_nmos = "../assets/template/nmos/nmos-training.sp"
    test_netlist_nmos = "../assets/template/nmos/nmos-testing-pro.sp"
    param_list_nmos = ["nmos-r", "nmos-w"]
    perform_list_nmos = ["nmos-bw", "nmos-pw", "nmos-a0"]
    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, -1, 1]
    if arguments_nmos is None:
        arguments_nmos = {
            "model_path": "../assets/model/45nm_CS.pm",
            "nmos-w_start": "2.88u",
            "nmos-w_stop": "6.63u",
            "nmos-w_change": "0.2u",
            "nmos-r_start": 620,
            "nmos-r_stop": 1450,
            "nmos-r_change": 5,
            "out": out_path
        }
    return Simulator(ngspice_exec, train_netlist_nmos, test_netlist_nmos, param_list_nmos, perform_list_nmos,
                     arguments_nmos, order, sign)


def cascode_circuit(arguments_cascode=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_cascade = "../assets/template/cascode/nmos-training-cascode.sp"
    test_netlist_cascade = "../assets/template/cascode/nmos-testing-cascode.sp"
    param_list_cascade = ["cascode-r", "cascode-w0", "cascode-w1"]
    perform_list_cascade = ["cascode-bw", "cascode-pw", "cascode-a0"]
    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, -1, 1]
    if arguments_cascode is None:
        arguments_cascode = {
            "model_path": "../assets/model/45nm_CS.pm",
            "cascode-w0_start": "4u",
            "cascode-w0_stop": "7.5u",
            "cascode-w0_change": "0.25u",
            "cascode-w1_start": "7u",
            "cascode-w1_stop": "10u",
            "cascode-w1_change": "0.2u",
            "cascode-r_start": 200,
            "cascode-r_stop": 500,
            "cascode-r_change": 18.75,
            "out": out_path
        }
    return Simulator(ngspice_exec, train_netlist_cascade, test_netlist_cascade, param_list_cascade,
                     perform_list_cascade,
                     arguments_cascode, order, sign)


def two_stage_circuit(arguments_two_stage=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_two_stage = "../assets/template/two-stage/TwoStageAmplifier.sp"
    test_netlist_two_stage = "../assets/template/two-stage/TwoStageAmplifier-Test.sp"
    param_list_two_stage = ["ts-w0", "ts-w1", "ts-w2"]
    perform_list_two_stage = ["ts-bw", "ts-pw", "ts-a0"]
    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, -1, 1]

    if arguments_two_stage is None:
        arguments_two_stage = {
            "model_path": "../assets/model/45nm_CS.pm",
            "ts-w0_start": "25u",
            "ts-w0_stop": "30u",
            "ts-w0_change": "0.5u",
            "ts-w2_start": "52u",
            "ts-w2_stop": "55.5u",
            "ts-w2_change": "0.5u",
            "ts-w1_start": "6u",
            "ts-w1_stop": "9u",
            "ts-w1_change": "0.5u",
            "out": out_path
        }
    simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
                                    perform_list_two_stage,
                                    arguments_two_stage, order, sign)
    simulator_two_stage.delete_existing_data = False
    return simulator_two_stage


def LNA_circuit(arguments_lna=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_lna = "../assets/template/LNA/LNA_Pooya.sp"
    test_netlist_lna = "../assets/template/LNA/LNA_Pooya.sp"
    param_list_lna = ["ls", "ld", "lg", "w"]
    perform_list_lna = ["Gt", "s11", "nf"]

    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, -1, -1]

    if arguments_lna is None:
        arguments_lna = {
            "model_path": "../assets/model/45nm_CS.pm",
            "ls_start": "747p",
            "ls_stop": "754p",
            "ls_change": "1p",
            "ld_start": "3.7n",
            "ld_stop": "4.4n",
            "ld_change": "0.1n",
            "lg_start": "9.4n",
            "lg_stop": "10.8n",
            "lg_change": "0.2n",
            "w_start": "73u",
            "w_stop": "76.5u",
            "w_change": "0.5u",
            "out": "../out/"
        }
    simulator_lna = Simulator(ngspice_exec, train_netlist_lna, test_netlist_lna, param_list_lna, perform_list_lna,
                              arguments_lna, order, sign)
    simulator_lna.delete_existing_data = True

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
            "vco-w_start": "4u",
            "vco-w_stop": "5.4u",
            "vco-w_change": "0.1u",
            "vco-w1_start": "2u",
            "vco-w1_stop": "10u",
            "vco-w1_change": "0.5u",
            "vco-w2_start": "15u",
            "vco-w2_stop": "17.8u",
            "vco-w2_change": "0.2u",
            "out": out_path
        }
    simulator_vco = Simulator(ngspice_exec, train_netlist_vco, test_netlist_vco, param_list_vco, perform_list_vco,
                              arguments_vco, order, sign)
    simulator_vco.delete_existing_data = False

    return simulator_vco


def Mixer_circuit(arguments_mixer=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_Mixer = ["../assets/mixer/conversiongain_mixer.sp", "../assets/mixer/new_distortion_mixer.sp",
                           "../assets/mixer/Power_consumption_mixer-test.sp"]
    test_netlist_Mixer = ["../assets/mixer/conversiongain_mixer-test.sp",
                          "../assets/mixer/new_distortion_mixer-test.sp",
                          "../assets/mixer/Power_consumption_mixer-test.sp"]
    param_list_Mixer = ["Vbrf", "RL", "WN", "WT"]
    perform_list_Mixer = ["conversiongain1", "mixer_distortion", "Power_consumption"]

    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, 1, 1]

    if arguments_mixer is None:
        arguments_mixer = {
            "model_path": "../assets/model/45nm_CS.pm",
            "Vbrf_start": "600m",
            "Vbrf_stop": "810m",
            "Vbrf_change": "30m",
            "RL_start": "240",
            "RL_stop": "520",
            "RL_change": "40",
            "WN_start": "8.55u",
            "WN_stop": "11.70u",
            "WN_change": "0.45u",
            "WT_start": "17.1u",
            "WT_stop": "23.4u",
            "WT_change": "0.9u",
            "out": "../out/"
        }
    simulator_Mixer = MultiSimulator(ngspice_exec, test_netlist_Mixer, test_netlist_Mixer, param_list_Mixer,
                                     perform_list_Mixer,
                                     arguments_mixer, order, sign)
    simulator_Mixer.save_error_log = True
    simulator_Mixer.delete_existing_data = True

    return simulator_Mixer


def Mixer_circuit(arguments_mixer=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_Mixer = ["../assets/mixer/conversiongain_mixer.sp", "../assets/mixer/new_distortion_mixer.sp",
                           "../assets/mixer/Power_consumption_mixer-test.sp"]
    test_netlist_Mixer = ["../assets/mixer/conversiongain_mixer-test.sp",
                          "../assets/mixer/new_distortion_mixer-test.sp",
                          "../assets/mixer/Power_consumption_mixer-test.sp"]
    param_list_Mixer = ["Vbrf", "RL", "WN", "WT"]
    perform_list_Mixer = ["conversiongain1", "mixer_distortion", "Power_consumption"]

    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, 1, 1]

    if arguments_mixer is None:
        arguments_mixer = {
            "model_path": "../assets/model/45nm_CS.pm",
            "Vbrf_start": "600m",
            "Vbrf_stop": "810m",
            "Vbrf_change": "30m",
            "RL_start": "240",
            "RL_stop": "520",
            "RL_change": "40",
            "WN_start": "8.55u",
            "WN_stop": "11.70u",
            "WN_change": "0.45u",
            "WT_start": "17.1u",
            "WT_stop": "23.4u",
            "WT_change": "0.9u",
            "out": "../out/"
        }
    simulator_Mixer = MultiSimulator(ngspice_exec, test_netlist_Mixer, test_netlist_Mixer, param_list_Mixer,
                                     perform_list_Mixer,
                                     arguments_mixer, order, sign)
    simulator_Mixer.save_error_log = True
    simulator_Mixer.delete_existing_data = True

    return simulator_Mixer


def Mixer_circuit(arguments_mixer=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_Mixer = ["../assets/mixer/conversiongain_mixer.sp", "../assets/mixer/new_distortion_mixer.sp",
                           "../assets/mixer/Power_consumption_mixer-test.sp"]
    test_netlist_Mixer = ["../assets/mixer/conversiongain_mixer-test.sp",
                          "../assets/mixer/new_distortion_mixer-test.sp",
                          "../assets/mixer/Power_consumption_mixer-test.sp"]
    param_list_Mixer = ["Vbrf", "RL", "WN", "WT"]
    perform_list_Mixer = ["conversiongain1", "mixer_distortion", "Power_consumption"]

    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, 1, 1]

    if arguments_mixer is None:
        arguments_mixer = {
            "model_path": "../assets/model/45nm_CS.pm",
            "Vbrf_start": "600m",
            "Vbrf_stop": "810m",
            "Vbrf_change": "30m",
            "RL_start": "240",
            "RL_stop": "520",
            "RL_change": "40",
            "WN_start": "8.55u",
            "WN_stop": "11.70u",
            "WN_change": "0.45u",
            "WT_start": "17.1u",
            "WT_stop": "23.4u",
            "WT_change": "0.9u",
            "out": "../out/"
        }
    simulator_Mixer = MultiSimulator(ngspice_exec, test_netlist_Mixer, test_netlist_Mixer, param_list_Mixer,
                                     perform_list_Mixer,
                                     arguments_mixer, order, sign)
    simulator_Mixer.save_error_log = True
    simulator_Mixer.delete_existing_data = True

    return simulator_Mixer


def Mixer_circuit(arguments_mixer=None, order=None, sign=None):
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_Mixer = ["../assets/mixer/conversiongain_mixer.sp", "../assets/mixer/new_distortion_mixer.sp",
                           "../assets/mixer/Power_consumption_mixer-test.sp"]
    test_netlist_Mixer = ["../assets/mixer/conversiongain_mixer-test.sp",
                          "../assets/mixer/new_distortion_mixer-test.sp",
                          "../assets/mixer/Power_consumption_mixer-test.sp"]
    param_list_Mixer = ["Vbrf", "RL", "WN", "WT"]
    perform_list_Mixer = ["conversiongain1", "mixer_distortion", "Power_consumption"]

    if order is None:
        order = [0, 2, 1]
    if sign is None:
        sign = [1, 1, 1]

    if arguments_mixer is None:
        arguments_mixer = {
            "model_path": "../assets/model/45nm_CS.pm",
            "Vbrf_start": "600m",
            "Vbrf_stop": "810m",
            "Vbrf_change": "30m",
            "RL_start": "240",
            "RL_stop": "520",
            "RL_change": "40",
            "WN_start": "8.55u",
            "WN_stop": "11.70u",
            "WN_change": "0.45u",
            "WT_start": "17.1u",
            "WT_stop": "23.4u",
            "WT_change": "0.9u",
            "out": "../out/"
        }
    simulator_Mixer = MultiSimulator(ngspice_exec, test_netlist_Mixer, test_netlist_Mixer, param_list_Mixer,
                                     perform_list_Mixer,
                                     arguments_mixer, order, sign)
    simulator_Mixer.save_error_log = True
    simulator_Mixer.delete_existing_data = True

    return simulator_Mixer
