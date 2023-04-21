import algo_test
from algo_test import *
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# 载入数据集

def PCA_2D():
    data = algo_test.get_data('CW_Data.csv')
    data = pre_processing(data, [0], duplicate=True, normalization_method='Min-Max')
    X_2d = algo_test.dimension_reduction(data, 'PCA', 2, label=6)
    # 使用PCA算法将数据降到2维
    # 绘制散点图
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title('2D PCA')
    plt.show()


def TSNE_2D():
    data = algo_test.get_data('CW_Data.csv')
    data = pre_processing(data, [0], duplicate=True, normalization_method='Min-Max')
    X_2d = algo_test.dimension_reduction(data, 'TSNE', 2, label=6)

    # 绘制散点图
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title('2D TSNE')
    plt.show()

def LDA_2D():
    data = algo_test.get_data('CW_Data.csv')
    data = pre_processing(data, [0], duplicate=True, normalization_method='Min-Max')
    X_2d = algo_test.dimension_reduction(data, 'LDA', 2, label=6)

    # 绘制散点图
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title('2D LDA')
    plt.show()

def Isomap_2D():
    data = algo_test.get_data('CW_Data.csv')
    data = pre_processing(data, [0], duplicate=True, normalization_method='Min-Max')
    X_2d = algo_test.dimension_reduction(data, 'Isomap', 2, label=6)

    # 绘制散点图
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title('2D Isomap')
    plt.show()

def MDS_2D():
    data = algo_test.get_data('CW_Data.csv')
    data = pre_processing(data, [0], duplicate=True, normalization_method='Min-Max')
    X_2d = algo_test.dimension_reduction(data, 'MDS', 2, label=6)

    # 绘制散点图
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title('2D MDS')
    plt.show()

def NMF_2D():
    data = algo_test.get_data('CW_Data.csv')
    data = pre_processing(data, [0], duplicate=True, normalization_method='Min-Max')
    X_2d = algo_test.dimension_reduction(data, 'NMF', 2, label=6)

    # 绘制散点图
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title('2D NMF')
    plt.show()


def knn_prediction():

    out_cm, out_fm, pred, X_train, X_test, y_train, y_pred, y_test, data, clf = algo_test.knn_plot()
    plt.figure()
    plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train, alpha=0.3)
    right = X_test[y_pred == y_test]
    wrong = X_test[y_pred != y_test]
    plt.scatter(x=right[:, 0], y=right[:, 1], color='r', marker='x', label='correct prediction')
    plt.scatter(x=wrong[:, 0], y=wrong[:, 1], color='r', marker='>', label='incorrect prediction')
    plt.legend(loc='best')
    plt.title('Visualization of KNN prediction')
    plt.show()


    # 画出决策边界
def plot_knn_decision_boundary(X, y, clf, k=5, resolution=0.1):
    """
    画出KNN分类器的决策边界
    :param X: 训练数据，numpy array，形状为(n_samples, n_features)
    :param y: 标签，numpy array，形状为(n_samples, )
    :param clf: KNN分类器的实例
    :param k: KNN算法中的K值，默认为5
    :param resolution: 决策边界的分辨率，默认为0.02
    """
    # 设置标记点的样式
    markers = ('s', 'x', 'o', '^', 'v')
    # 设置颜色
    colors = ('red', 'blue', 'green', 'yellow', 'cyan')
    # 设置颜色映射
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 画出决策边界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # 画出样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.title(f'KNN (k={k}) decision boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.show()



PCA_2D()
TSNE_2D()
knn_prediction()

out_cm, out_fm, pred, X_train, X_test, y_train, y_pred, y_test, data, clf = algo_test.knn_plot()
plot_knn_decision_boundary(X_train, y_train, clf, k=5, resolution=0.1)
