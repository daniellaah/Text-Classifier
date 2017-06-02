import os
import sys
import time
import random
import re
import jieba
from sklearn.naive_bayes import BernoulliNB
from sklearn.externals import joblib

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
    if len(sys.argv) == 2:
        save_path = sys.argv[1]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        start_time = time.time()
        train_data = words_extract('train_test_data/train')
        test_data = words_extract('train_test_data/test')
        feature_words = get_feature_words(train_data, size=1000, stopwords_file="stopwords.txt")
        X_train, y_train, X_test, y_test = train_test_extract(train_data, test_data, feature_words)
        print("数据集构造用时%ss." % str(time.time()-start_time))

        start_time = time.time()
        clf = BernoulliNB().fit(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        print("训练用时%ss" % (str(time.time()-start_time)))
        print("精度为%ss" % str(test_accuracy))

        if not os.path.exists(os.path.join(save_path, 'Bernoulli_NaiveBayes.pkl')):
            joblib.dump(clf, os.path.join(save_path, 'Bernoulli_NaiveBayes.pkl'))

        if not os.path.exists(os.path.join(save_path, 'feature_words.txt')):
            with open(os.path.join(save_path, "feature_words.txt"), 'w') as f:
                for word in feature_words:
                    f.write(word + '\n')
    else:
        print("参数错误")
