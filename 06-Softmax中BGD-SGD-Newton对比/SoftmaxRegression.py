import os
import sys
import time
import random
import re
import jieba
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class SoftmaxRegression():
    def __init__(self, k, class_list):
        self.weight = np.array([], dtype=float) # k * n + 1
        self.k = k
        self.int_to_label = {}
        self.label_to_int = {}
        for i, label in enumerate(class_list):
            self.int_to_label[i] = label
            self.label_to_int[label] = i

    def softmax(self, X):
        matrix = np.exp(np.dot(self.weight, X.T))
        return matrix / np.sum(matrix, axis=0)

    def predict(self, X):
        X = np.array(X)
        sample_nums = X.shape[0]
        X = np.column_stack((np.ones(sample_nums), X))
        int_result = np.argmax(self.softmax(X), axis=0)
        label_result = []
        for i in range(len(int_result)):
            label_result.append(self.int_to_label[int_result[i]])
        return label_result

    def loss_func(self, X, Y, reg):
        sample_nums, feature_nums = X.shape[0], X.shape[1]
        loss = -np.sum(Y * np.log(self.softmax(X)))/sample_nums
        regularization = np.sum(self.weight * self.weight) * reg / 2
        return loss +  regularization

    def fit_BGD(self, X, y, alpha=0.01, reg=0.1, max_iter=1000, epsilon=1e-10):
        X = np.array(X)
        sample_nums, feature_nums = X.shape[0], X.shape[1] + 1
        Y = np.zeros((self.k, sample_nums))
        for i, label in enumerate(y):
            Y[self.label_to_int[label], i] = 1
        X = np.column_stack((np.ones(sample_nums), X))
        self.weight = np.zeros((self.k, feature_nums), dtype=float)
        for i in range(max_iter):
            batch_gradient = np.dot((Y - self.softmax(X)), X)  / sample_nums
            self.weight += (alpha * batch_gradient - reg * self.weight)
        return self

    def fit_SGD(self, X, y, alpha=0.01, reg=0.1, max_iter=1000):
        X = np.array(X)
        sample_nums, feature_nums = X.shape[0], X.shape[1] + 1
        Y = np.zeros((self.k, sample_nums))
        for i, label in enumerate(y):
            Y[self.label_to_int[label], i] = 1
        X = np.column_stack((np.ones(sample_nums), X))
        self.weight = np.zeros((self.k, feature_nums), dtype=float)
        count = (max_iter * sample_nums)/ 10
        loss = self.loss_func(X, Y, reg)
        for i in range(max_iter):
            for j in range(sample_nums):
                rand = np.random.randint(sample_nums)
                stochasitc_gradient = np.dot((Y[:, rand] - self.softmax(X[rand, :])).reshape(self.k, 1),
                                                X[rand, :].reshape(1, feature_nums))
                self.weight += (alpha * stochasitc_gradient - reg * self.weight)
        print('iter nums: %s' % str(i + 1))
        print('loss: %s' % str(self.loss_func(X, Y, reg)))
        return self

    def fit_Newton(self, X, y, reg=0.01, max_iter=10):
        X = np.array(X)
        sample_nums, feature_nums = X.shape[0], X.shape[1] + 1
        Y = np.zeros((self.k, sample_nums))
        for i, label in enumerate(y):
            Y[self.label_to_int[label], i] = 1
        X = np.column_stack((np.ones(sample_nums), X))
        self.weight = np.zeros((self.k, feature_nums), dtype=float)
        for i in range(max_iter):
            matrix = np.exp(np.dot(self.weight, X.T))
            sum_matrix = np.sum(matrix, axis=0)
            gradient = -np.dot((Y - self.softmax(X)), X)  / sample_nums
            for j in range(self.k):
                hessian = np.dot(X.T, np.dot((sum_matrix - matrix[j]) / (sum_matrix * sum_matrix), matrix[j]) * X)
                self.weight[j] -= np.dot(np.linalg.inv(hessian), gradient[j])
        print('iter nums: %s' % str(i + 1))
        print('loss: %s' % str(self.loss_func(X, Y, reg)))
        return self

    def score(self, X, y_true):
        return accuracy_score(y_true, self.predict(X))

def words_extract(news_folder):
    """从所有文件内容提取词
    Args:
        news_folder/
            财经/
            体育/
            娱乐/
    """
    subfolder_list = [subfolder for subfolder in os.listdir(news_folder) \
                        if os.path.isdir(os.path.join(news_folder, subfolder))]
    data_list = [] # element: ([word1, word2, ...], "财经")

    jieba.enable_parallel(4)
    # 遍历每个类别下的新闻
    for subfolder in subfolder_list:
        news_class = subfolder
        subfolder = os.path.join(news_folder, subfolder)
        news_list = [os.path.join(subfolder, news) for news in os.listdir(subfolder) \
                        if os.path.isfile(os.path.join(subfolder, news))]
        for news in news_list:
            with open(news, 'r') as f:
               content = f.read()
            word_list = jieba.lcut(content)
            data_list.append((word_list,news_class)) # element: ([word1, word2, ...], "财经")
    jieba.disable_parallel()
    return data_list

def get_stopwords(stopwords_file="stopwords.txt"):
    """返回所有停止词
    Args:
        stopwords_file: 停止词文件路径
    """
    stopwords_set = set()
    with open(stopwords_file, 'r') as f:
        for line in f.readlines():
            stopwords_set.add(line.strip())
    return stopwords_set

def get_feature_words(train_data_list, size=1000, stopwords_file="stopwords.txt"):
    """从训练集提取待选特征词
    Args:
        train_data_list:
            element: ([word1, word2, ...], "财经")
        stopwords_file: 停止词文件路径
    """
    stopwords = get_stopwords(stopwords_file)
    feature_words_dict = {}
    for element in train_data_list:
        for word in element[0]:
            if not re.match("[a-z0-9A-Z]", word) and len(word) > 1 and word not in stopwords:
                if word in feature_words_dict:
                    feature_words_dict[word] += 1
                else:
                    feature_words_dict[word] = 1
    feature_words_tuple = sorted(feature_words_dict.items(), key=lambda x:x[1], reverse=True)
    feature_words = list(list(zip(*feature_words_tuple))[0])
    return feature_words[:size] if len(feature_words) > size else feature_words

def train_test_extract(train_data, test_data, feature_words):
    """从训练数据与测试数据提取 X_train, y_train, X_test, y_test
    Args:
        train_data: 训练数据
        test_data: 测试数据
        feature_words: 特征词
    """
    X_train = [[1 if word in element[0] else 0 for word in feature_words] for element in train_data]
    y_train = [element[1] for element in train_data]
    X_test = [[1 if word in element[0] else 0 for word in feature_words] for element in test_data]
    y_test = [element[1] for element in test_data]
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    if not (os.path.exists("X_train.csv") and
        os.path.exists("y_train.csv") and
        os.path.exists("X_test.csv") and
        os.path.exists("y_test.csv")):
        start_time = time.time()
        train_data = words_extract('train_test_data/train')
        test_data = words_extract('train_test_data/test')
        feature_words = get_feature_words(train_data, size=1000, stopwords_file="stopwords.txt")
        X_train, y_train, X_test, y_test = train_test_extract(train_data, test_data, feature_words)
        print("数据集构造用时%ss." % str(time.time()-start_time))
        np.savetxt("X_train.csv", X_train, fmt='%i')
        np.savetxt("X_test.csv", X_test, fmt='%s')

        with open("y_train.csv", 'w') as f_obj:
            for label in y_train:
                f_obj.write(label + '\n')

        with open("y_test.csv", 'w') as f_obj:
            for label in y_test:
                f_obj.write(label + '\n')
    else:
        print("数据集已经存在, 直接读取...")
        start_time = time.time()
        X_train = np.genfromtxt("X_train.csv")
        X_test = np.genfromtxt("X_test.csv")
        y_train, y_test = [], []

        with open("y_train.csv", 'r') as f_obj:
            y_train = f_obj.read().strip().split('\n')

        with open("y_test.csv", 'r') as f_obj:
            y_test = f_obj.read().strip().split('\n')

        print("读取用时: %ss" % str(time.time()-start_time))

    class_list = ['IT', '娱乐', '财经', '体育']

    print('-------------------Batch Gradient Descent---------------------------')
    start_time = time.time()
    clf_BGD = SoftmaxRegression(len(class_list), class_list).fit_BGD(X_train, y_train, alpha=0.1, reg=0.01, max_iter=2000, epsilon=0.0)
    print("训练用时: %ss" % (str(time.time()-start_time)))
    start_time = time.time()
    test_accuracy = clf_BGD.score(X_test, y_test)
    print("测试用时: %ss" % (str(time.time()-start_time)))
    print("准确率为: %s" % str(test_accuracy))

    print('-----------------Stochasitc Gradient Descent------------------------')
    start_time = time.time()
    clf_SGD = SoftmaxRegression(len(class_list), class_list).fit_SGD(X_train, y_train, alpha=0.01, reg=0.001, max_iter=3)
    print("训练用时: %ss" % (str(time.time()-start_time)))
    start_time = time.time()
    test_accuracy = clf_SGD.score(X_test, y_test)
    print("测试用时: %ss" % (str(time.time()-start_time)))
    print("准确率为: %s" % str(test_accuracy))

    print('------------------------Newton Method-------------------------------')
    start_time = time.time()
    clf_Newton = SoftmaxRegression(len(class_list), class_list).fit_Newton(X_train, y_train, reg=0.0, max_iter=1)
    print("训练用时: %ss" % (str(time.time()-start_time)))
    start_time = time.time()
    test_accuracy = clf_Newton.score(X_test, y_test)
    print("测试用时: %ss" % (str(time.time()-start_time)))
    print("准确率为: %s" % str(test_accuracy))


    # GridSearch
    # alpha_list = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    # reg_list = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    # for alpha in alpha_list:
    #     for reg in reg_list:
    #         start_time = time.time()
    #         clf_SGD = SoftmaxRegression(len(class_list), class_list).fit_SGD(X_train, y_train, alpha=alpha, reg=reg, max_iter=20)
    #         test_accuracy = clf_SGD.score(X_test, y_test)
    #         print("训练用时%ss" % (str(time.time()-start_time)))
    #         print("alpha=%s, lambda=%s, 精度为%s" % (str(alpha), str(reg), str(test_accuracy)))
    #         print('---------------------------------------------------')
