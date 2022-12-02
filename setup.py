import os

def create_folder(path, folder_name):
    if os.path.exists(path):
        print("{} FOLDER EXIST ALREADY".format(folder_name))
    else:
        os.mkdir(path)
        print("{} FOLDER DOES NOT EXIST, CREATING NOW".format(folder_name))

def setup():
    '''
    RUN THIS FUNCTION TO SETUP REQUIRED FOLDER FOR RUNNING PIPELINE
    '''

    pwd = os.getcwd()


    simulate_out_folder = os.path.join(pwd, "out")
    train_result_folder = os.path.join(pwd, "result_out")
    train_plot_folder = os.path.join(pwd, "out_plot")
    tmp_netlist_out_folder = os.path.join(pwd, "tmp_out")
    test_data_out_folder = os.path.join(pwd, "data")

    nmos_data_out_path = os.path.join(test_data_out_folder, "nmos")
    cascode_data_out_path = os.path.join(test_data_out_folder, "cascode")
    LNA_data_out_path = os.path.join(test_data_out_folder, "LNA")
    mixer_data_out_path = os.path.join(test_data_out_folder, "mixer")
    two_stage_data_out_path = os.path.join(test_data_out_folder, "two_stage")
    VCO_data_out_path = os.path.join(test_data_out_folder, "VCO")

    create_folder(simulate_out_folder, "SIMULATE OUT")
    create_folder(train_result_folder, "TRAIN RESULT")
    create_folder(train_plot_folder, "TRAIN PLOT")
    create_folder(tmp_netlist_out_folder, "TEMP NETLIST OUT")
    create_folder(test_data_out_folder, "TEST DATA OUT")
    create_folder(nmos_data_out_path, "NMOS DATA OUT")
    create_folder(cascode_data_out_path, "CASCODE DATA OUT")
    create_folder(LNA_data_out_path, "LNA DATA OUT")
    create_folder(mixer_data_out_path, "MIXER DATA OUT")
    create_folder(two_stage_data_out_path, "TWO STAGE DATA OUT")
    create_folder(VCO_data_out_path, "VCO DATA OUT")


if __name__ == '__main__':
    setup()