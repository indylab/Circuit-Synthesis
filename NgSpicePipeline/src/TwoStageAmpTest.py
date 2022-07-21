import numpy as np
import pandas as pd
import os

from Simulator import Simulator

if __name__ == '__main__':
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_two_stage = "../assets/TwoStageAmplifier.sp"
    test_netlist_two_stage = "../assets/TwoStageAmplifier-Test.sp"
    param_list_two_stage = ["w0", "w1", "w2"]
    perform_list_two_stage = ["bw", "pw", "a0"]

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
        "w1_change": "1u",
        "out": "../out/"
    }
    simulator_two_stage = Simulator(ngspice_exec, train_netlist_two_stage, test_netlist_two_stage, param_list_two_stage,
                                    perform_list_two_stage,
                                    arguments_two_stage)
    simulator_two_stage.delete_existing_data = True
    x, y = simulator_two_stage.run_training()
    data = np.hstack((x, y))
    pd.DataFrame(data).to_csv(os.path.join(arguments_two_stage["out"], "TwoStageAmpData.csv"))

    x_pred, y_pred = simulator_two_stage.runSimulation(x)

    print(data.shape)
    print(x.shape,y.shape)
    for i, p in enumerate(data):
        print("=" * 100)
        print(f"point: {p}\n", f"x sim: {x_pred[i]}\n", f"y sim: {y_pred[i]}")
