from NgSpicePipeline.src.Simulator import Simulator
import numpy as np

if __name__ == '__main__':
    ngspice_exec = "../../ngspice/Spice64/bin/ngspice.exe"
    train_netlist_lna = "../assets/LNA.sp"
    test_netlist_lna = "../assets/LNA_test"
    param_list_lna = ["ls", "ld", "lg", "r", "w"]
    perform_list_lna = ["Gmax", "Gp", "s11", "nf"]
    arguments_lna = {
        "model_path": "../assets/45nm_CS.pm",
        "ls_start": "58.3p",
        "ls_stop": "60.8p",
        "ls_change": "1p",
        "ld_start": "4.4n",
        "ld_stop": "5.4n",
        "ld_change": "1n",
        "lg_start": "14.8n",
        "lg_stop": "15.8n",
        "lg_change": "0.50n",
        "r_start": "800",
        "r_stop": "1050",
        "r_change": "50",
        "w_start": "51u",
        "w_stop": "53u",
        "w_change": "0.4u",
        "out": "../out/"
    }
    simulator_lna = Simulator(ngspice_exec, train_netlist_lna, test_netlist_lna, param_list_lna, perform_list_lna,
                              arguments_lna)
    simulator_lna.delete_existing_data = True
    param, perform = simulator_lna.run_training()
    param_sim, perform_sim = simulator_lna.run_training() #simulator_lna.runSimulation(parameters=param)
    perform_sim = np.array(perform_sim)
    perform = np.array(perform)
    print("sim\t real")
    count = 0
    print("="*5,"Perform","="*5)
    for i in range(perform_sim.shape[0]):
        cond = np.allclose(perform_sim[i, :], perform[i, :])
        if cond:
            count += 1
        print(i,perform_sim[i, :], perform[i, :], cond)
    print(count, "/", perform.shape[0])
    print("=" * 5, "End Perform", "=" * 5)
    print("=" * 5, "param", "=" * 5)
    count = 0
    for i in range(param_sim.shape[0]):
        cond = np.allclose(param_sim[i, :], param[i, :])
        if cond:
            count += 1
        print(i,param_sim[i, :], param[i, :], cond)
    print("=" * 5, "end param", "=" * 5)
    print(count, "/", param.shape[0])
    assert (np.allclose(perform_sim, perform)), "failed"
    print("success")
