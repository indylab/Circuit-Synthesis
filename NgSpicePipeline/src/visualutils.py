import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
import numpy as np
import seaborn as sns

def plot_parameter(X, Y, reduce_dim_x = True, reduce_dim_y = True):

    assert X.shape[1] == 2 or reduce_dim_x
    assert Y.shape[1] == 2 or reduce_dim_y

    embedded_X = X
    embedded_Y = Y
    if reduce_dim_x and X.shape[1] > 2:
        #embedded_X = TSNE(n_components=2, init='random').fit_transform(X)
        #embedded_X = LocallyLinearEmbedding(n_components=2).fit_transform(X)
        embedded_X = MDS(n_components=2).fit_transform(X)
    if reduce_dim_y and Y.shape[1] > 2:
        #embedded_Y = TSNE(n_components=2, init='random').fit_transform(Y)
        #embedded_Y = LocallyLinearEmbedding(n_components=2).fit_transform(Y)
        embedded_Y = MDS(n_components=2).fit_transform(Y)

    fig, axes = plt.subplots(figsize=(36, 12))
    axes.get_xaxis().set_visible(False)  # remove erroreas graph axis
    axes.get_yaxis().set_visible(False)

    C = np.ones((X.shape[0], 1, 3))

    for i,c in enumerate(C):
        c[:,1] = embedded_X[i,0] / embedded_X[:, 0].max()
        c[:,2] = embedded_X[i,1] / embedded_X[:, 1].max()
    C = C.reshape(-1, 3)
    fig.add_subplot(121)
    plt.scatter(embedded_X[:, 0], embedded_X[:, 1], c= C)
    fig.add_subplot(122)
    plt.scatter(embedded_Y[:, 0], embedded_Y[:, 1], c= C)
    plt.show()


def get_margin_error(y_hat, y, sign=None):
    sign = np.array(sign)
    if sign is not None:
        y_hat = y_hat * sign
        y = y * sign

    greater = np.all(y_hat >= y, axis=1)

    a_err = (np.abs(y_hat - y))
    err = np.divide(a_err, y, where=y != 0)
    max_err = np.max(err, axis=1)
    max_err[greater] = 0

    return max_err

def graph_margin(margin_error, margins, percentage = False):
    counts = []
    margin_error = np.array(margin_error)
    for margin in margins:
        if percentage:
            counts.append((margin_error <= margin).sum() / len(margin_error))
        else:
            counts.append((margin_error <= margin).sum())

    sns.lineplot(x = margins, y = counts)
    if percentage:
        plt.ylim(0,1.2)

    plt.xlim(0.5, 0)
    plt.show()