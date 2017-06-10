import os
import sys
import time
import random
import re
import jieba
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

        print("读取用时%ss." % str(time.time()-start_time))

    class_list = ['IT', '娱乐', '财经', '体育']

    print('-------------------Linear SVM---------------------------')
    start_time = time.time()
    clf = SVC(C='0.03', kernel='linear').fit(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("训练用时%ss" % (str(time.time()-start_time)))
    print("精度为%s" % str(test_accuracy))
    # c_list = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    # for c in c_list:
    #     start_time = time.time()
    #     clf = SVC(C=c, kernel='linear').fit(X_train, y_train)
    #     test_accuracy = clf.score(X_test, y_test)
    #     print("训练用时%ss" % (str(time.time()-start_time)))
    #     print("c=%s, 精度为%s" % (str(c), str(test_accuracy)))
    #     print('---------------------------------------------------')

    print('-------------------RBF SVM---------------------------')
    start_time = time.time()
    clf = SVC(C='0.03', kernel='rbf').fit(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("训练用时%ss" % (str(time.time()-start_time)))
    print("精度为%s" % str(test_accuracy))

    # c_list = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    # for c in c_list:
    #     start_time = time.time()
    #     clf = SVC(C=c, kernel='linear').fit(X_train, y_train)
    #     test_accuracy = clf.score(X_test, y_test)
    #     print("训练用时%ss" % (str(time.time()-start_time)))
    #     print("c=%s, 精度为%s" % (str(c), str(test_accuracy)))
    #     print('---------------------------------------------------')
