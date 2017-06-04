import os
import sys
import time
import random
import re
import jieba
import numpy as np
from sklearn.metrics import accuracy_score

def sigmoid(z):
       return 1.0 / (1 + np.exp(-z))

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

    def fit(self, X, y, alpha=0.01, reg=0.3, iter_nums=1000):
        X = np.array(X)
        sample_nums, feature_nums = X.shape[0], X.shape[1] + 1
        Y = np.zeros((self.k, sample_nums))
        for i, label in enumerate(y):
            Y[self.label_to_int[label], i] = 1
        X = np.column_stack((np.ones(sample_nums), X))
        self.weight = np.zeros((self.k, feature_nums), dtype=float)
        for i in range(iter_nums):
            self.weight += (alpha * np.dot((Y - self.softmax(X)), X) - reg * self.weight)
        return self

    def score(self, X, y_true):
        return accuracy_score(y_true, self.predict(X))

class LogisticRegression():
    def __init__(self):
        self.weight = np.array([], dtype=float)

    def fit(self, X, y, alpha=0.01, iter_nums=1000):
        X = np.array(X)
        sample_nums, feature_nums = X.shape[0], X.shape[1] + 1
        y = np.array(y).reshape(sample_nums, 1)
        X = np.column_stack((np.ones(sample_nums), X))
        init_weight = np.zeros(feature_nums, dtype=float).reshape(feature_nums, 1)
        for i in range(iter_nums):
            h = sigmoid(np.dot(X, init_weight))
            init_weight += alpha * np.dot(X.T, (y - h))
        self.weight = init_weight
        return self

    def predict(self, X):
        X = np.array(X)
        sample_nums = X.shape[0]
        X = np.column_stack((np.ones(sample_nums), X))
        result = []
        for sample in X:
            if sigmoid(np.dot(sample, self.weight)) >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return result

    def score(self, X, y_true):
        y_predict = self.predict(X)
        return accuracy_score(y_predict, y_true)

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
    start_time = time.time()
    train_data = words_extract('train_test_data_6000/train')
    test_data = words_extract('train_test_data_6000/test')
    feature_words = get_feature_words(train_data, size=1000, stopwords_file="stopwords.txt")
    X_train, y_train, X_test, y_test = train_test_extract(train_data, test_data, feature_words)
    print("数据集构造用时%ss." % str(time.time()-start_time))

    class_list = ['IT', '娱乐', '财经', '体育']
    start_time = time.time()
    clf = SoftmaxRegression(len(class_list), class_list).fit(X_train, y_train, alpha=0.00001, reg=0.0, iter_nums=1000)
    test_accuracy = clf.score(X_test, y_test)
    print("训练用时%ss" % (str(time.time()-start_time)))
    print("精度为%s" % str(test_accuracy))
