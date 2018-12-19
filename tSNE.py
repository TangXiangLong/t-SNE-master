import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.manifold import Isomap
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import decomposition

def loadmnist():
    '读取mnist数据集'''
    mnist = input_data.read_data_sets('./MNIST', one_hot=True)
    X = mnist.validation.images
    labels = mnist.validation.labels
    y = np.argmax(labels, axis=1)
    return X, y

def visualize(X,y):
    '嵌入空间可视化'''
    x_min, x_max = X.min(0), X.max(0)
    X_norm = (X - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./images/%s.jpg"%(sys._getframe().f_back.f_code.co_name))
    plt.show()

def tSNE():
    X, y = loadmnist()
    X_tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, n_iter=1000, verbose=1).fit_transform(X)
    visualize(X_tsne, y)

def isomap():
    X,y = loadmnist()
    X_isomap = Isomap(n_components=2).fit_transform(X)
    visualize(X_isomap, y)

def LLE():
    X,y = loadmnist()
    X_LLE = LocallyLinearEmbedding(n_components=2).fit_transform(X)
    visualize(X_LLE, y)

def PCA():
    X,y = loadmnist()
    X_PCA = decomposition.PCA(n_components=2).fit_transform(X)
    visualize(X_PCA, y)


def main(argv=None):
    tSNE()
    isomap()
    LLE()
    PCA()

if __name__ == "__main__":
    main()