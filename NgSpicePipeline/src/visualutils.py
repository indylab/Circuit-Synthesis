import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
import numpy as np

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