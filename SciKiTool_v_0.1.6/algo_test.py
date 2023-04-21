import numpy as np
import pandas as pd
from matplotlib.testing.jpl_units import km
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import Isomap, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, silhouette_score, \
    davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score


def get_data(path):
    data = pd.read_csv(path, sep=',', header=0)
    data = pd.DataFrame(data)  # 转换为DataFrame格式
    data = data.dropna()  # 去除空值
    return data


def pre_processing(data, drop_useless, duplicate=False, normalization_method=''):
    # 数据预处理, duplicate:是否去除重复值, normalization_method:标准化方法
    if duplicate:
        data = data.drop_duplicates()  # 去除重复值
    if drop_useless != -1:
        data = data.drop(columns=data.columns[drop_useless])  # 去除无用列
    if normalization_method == 'Z-Score':  # 标准化
        data_scalar = preprocessing.StandardScaler()
        data = data_scalar.fit_transform(data)
    elif normalization_method == 'Min-Max':  # 归一化
        data_scalar = preprocessing.MinMaxScaler()
        data = data_scalar.fit_transform(data)
    return data


def dimension_reduction(data, method, n_components, label=-1):  # 数据降维, method:降维方法, n_components:降维后的维度, lable:是否有标签
    if method == 'PCA':
        data = PCA(n_components=n_components).fit_transform(data)
    elif method == 'LDA' and label != -1:
        data = LDA(n_components=n_components).fit_transform(data, label)  # y为标签, 且必须
    elif method == 'TSNE':
        data = TSNE(n_components=n_components).fit_transform(data)
    elif method == 'ISOmap':
        data = Isomap(n_components=n_components).fit_transform(data)
    elif method == 'NMF':
        data = NMF(n_components=n_components).fit_transform(data)
    elif method == 'MDS':
        data = MDS(n_components=n_components).fit_transform(data)
    return data


def decision_tree(data, y):  # 决策树
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred


def random_forest(data, y):  # 随机森林分类器
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred


def svm(data, y):  # 支持向量机分类器
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred


def knn(data, y):  # KNN分类器
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred

def knn_plot():
    label = 6  # 标签所在列，需手动设置
    uesless = [0]  # 无用列，需手动设置
    useless_columns = uesless + [label]
    print(useless_columns)
    data = get_data('CW_Data.csv')
    y = data.iloc[:, label]
    data = pre_processing(data, drop_useless=useless_columns, duplicate=False, normalization_method='')
    data = dimension_reduction(data, 'TSNE', 2, label=6)

    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred, X_train, X_test, y_train, y_test, y_pred, data, clf

def naive_bayesian(data, y):  # 朴素贝叶斯分类器
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred


def logistic_regression(data, y):  # 逻辑回归分类器
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred


def neural_network(data, y):  # 神经网络分类器
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred


def four_metrics(y_test, y_pred):  # 评价指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return pd.DataFrame([[accuracy, precision, recall, f1]], columns=['accuracy', 'precision', 'recall', 'f1'])


def supervised_learning(data, y, method):  # 有监督学习, method:分类器
    if method == 'SVM':
        return svm(data, y)
    elif method == 'Decision_Tree':
        return decision_tree(data, y)
    elif method == 'Random_Forest':
        return random_forest(data, y)
    elif method == 'KNN':
        return knn(data, y)


def kmeans(data, n_clusters):  # k-means 无监督分类器
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(data)  # n_clusters:聚类数
    y_pred = k_means.predict(data)
    center = k_means.cluster_centers_  # 聚类中心
    return y_pred, unsupervised_score(data, y_pred), center


def dbscan(data, eps=0.5, min_samples=5):  # DBSCAN 无监督分类器
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)  # eps:半径, min_samples:最小样本数
    y_pred = db.labels_
    return y_pred, unsupervised_score(data, y_pred)


def unsupervised_score(data, y_pred):  # 无监督学习评价指标
    silhouette_coefficient = silhouette_score(data, y_pred)  # 轮廓系数, 越大越好
    calinski_harabasz = calinski_harabasz_score(data, y_pred)  # Calinski-Harabasz指数, 越大越好
    davies_bouldin = davies_bouldin_score(data, y_pred)  # Davies-Bouldin指数, 越小越好
    return pd.DataFrame([[silhouette_coefficient, calinski_harabasz, davies_bouldin]],
                        columns=['silhouette_coefficient', 'calinski_harabasz', 'davies_bouldin'])


def unsupervised_learning(data, method, *args):  # 无监督学习, method:分类器
    if method == 'K-Means':
        return kmeans(data, *args)
    elif method == 'DBSCAN':
        return dbscan(data, *args)



def run():
    label = 6  # 标签所在列，需手动设置
    uesless = [0]  # 无用列，需手动设置
    useless_columns = uesless + [label]
    print(useless_columns)
    data = get_data('CW_Data.csv')
    y = data.iloc[:, label]
    data = pre_processing(data, drop_useless=useless_columns, duplicate=False, normalization_method='')
    data = dimension_reduction(data, 'TSNE', 2, label=6)
    print(data)
    out_cm, out_fm, pred = knn(data, y)
    # print(out_cm, '\n', out_fm, '\n', out_pred)
    # pred, score = unsupervised_learning(data, 'DBSCAN', 1, 4)
    print(pred, '\n', out_fm)
    print(out_cm)


if __name__ == '__main__':
    run()
