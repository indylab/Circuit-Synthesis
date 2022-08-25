import pickle

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filehandler = open("../assets/TwoStageAmplifiersize_tests", 'rb')
    object = pickle.load(filehandler)
    means_d = dict()
    errs_d = dict()
    lows = dict()
    highs = dict()
    print(object.keys())
    for k, v in object.items():
        print(np.array(v))
        object[k] = np.array(v).mean(axis=0)
        means_d[k] = np.array(v).mean(axis=0)
        lows[k] = means_d[k] - np.min(np.array(v),axis=0)
        highs[k] = np.max(np.array(v),axis=0) -  means_d[k]

    cir_name = "Two Stage Amplifier"
    trials = 5
    acc_lvl = 0
    accs = [1, 5, 10]
    means = [x[acc_lvl] for x in means_d.values()]
    highs = [x[acc_lvl] for x in highs.values()]
    lows = [x[acc_lvl] for x in lows.values()]
    big_errors = np.vstack((lows,highs))
    x_pos = np.arange(len(object.keys()))
    fig, ax = plt.subplots()
    print(big_errors)
    ax.bar(x_pos, np.array(means)*100, yerr=big_errors*100, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(f'% Success rate at {accs[acc_lvl]}% accuracy')
    ax.set_xlabel(f'Training Test Size')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(object.keys())
    ax.set_title(f'{cir_name}: Training Set Size Tests at {accs[acc_lvl]}% accuracy ({trials} trials)')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()
