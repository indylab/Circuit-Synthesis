import os


def setup():
    '''
    RUN THIS FUNCTION TO SETUP REQUIRED FOLDER FOR RUNNING PIPELINE
    '''

    pwd = os.getcwd()

    parent_root = os.path.join(pwd, os.pardir)

    simulate_out_folder = os.path.join(parent_root, "out")
    train_result_folder = os.path.join(parent_root, "result_out")
    train_plot_folder = os.path.join(parent_root, "out_plot")
    tmp_netlist_out_folder = os.path.join(os.path.join(parent_root, "assets"), "tmp_out")

    if os.path.exists(simulate_out_folder):
        print("SIMULATE OUT FOLDER EXIST ALREADY")
    else:
        print("SIMULATE OUT FOLDER DOES NOT EXIST, CREATING SIMULATE OUT FOLDER")
        os.mkdir(simulate_out_folder)

    if os.path.exists(train_result_folder):
        print("TRAIN RESULT FOLDER EXIST ALEADY")
    else:
        print("TRAIN RESULT FOLDER DOES NOT EXIST, CREATING TRAIN RESULT FOLDER")
        os.mkdir(train_result_folder)

    if os.path.exists(train_plot_folder):
        print("TRAIN PLOT FOLDER EXIST ALREADY")
    else:
        print("TRAIN PLOT FOLDER DOES NOT EXIST, CREATING TRAIN PLOT FOLDER")
        os.mkdir(train_plot_folder)

    if os.path.exists(tmp_netlist_out_folder):
        print("TMP NETLIST OUT FOLDER EXIST ALREADY")
    else:
        print("TMP NETLIST OUT FOLDER DOES NOT EXIST, CREATING TMP NETLIST OUT FOLDER")
        os.mkdir(tmp_netlist_out_folder)








if __name__ == '__main__':
    setup()