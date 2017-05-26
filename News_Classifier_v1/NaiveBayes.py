import os
import time
import random
import re
import jieba
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def get_feature_words(news_folder, size=1000, stopwords_file="stopwords.txt"):
    """从所有文件内容提取特征词
    Args:
        news_folder/
            财经/
            体育/
            娱乐/
    """
    news_classes = [subfolder for subfolder in os.listdir(news_folder) \
                        if os.path.isdir(os.path.join(news_folder, subfolder))]
    stopwords = get_stopwords(stopwords_file)
    feature_words_dict = {}
    # 遍历每个类别下的新闻
    jieba.enable_parallel(4)
    for news_class in news_classes:
        subfolder = os.path.join(news_folder, news_class)
        news_list = [os.path.join(subfolder, news) for news in os.listdir(subfolder) \
                        if os.path.isfile(os.path.join(subfolder, news))]
        for news in news_list:
            with open(news, 'r') as f:
                content = f.read()
                word_list = jieba.lcut(content)
                for word in word_list:
                    if not re.match("[a-z0-9A-Z]", word) and len(word) > 1 and word not in stopwords:
                        if word in feature_words_dict:
                            feature_words_dict[word] += 1
                        else:
                            feature_words_dict[word] = 1
    jieba.disable_parallel()
    feature_words_tuple = sorted(feature_words_dict.items(), key=lambda x:x[1], reverse=True)
    feature_words = list(list(zip(*feature_words_tuple))[0])
    return set(feature_words[:size]) if len(feature_words) > size else set(feature_words)

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

def get_probability(news_folder, feature_words):
    """从所有文件内容提取词
    Args:
        news_folder/
            财经/
            体育/
            娱乐/
    """
    news_classes = [subfolder for subfolder in os.listdir(news_folder) \
                        if os.path.isdir(os.path.join(news_folder, subfolder))]
    data_list = [] # element: ([word1, word2, ...], "财经")
    prob_matrix = pd.DataFrame(index=feature_words, columns=news_classes)
    prob_classes = {}
    for cls in news_classes:
        prob_classes[cls] = 0.0
    num_of_all_news = 0
    prob_count = {}
    for word in feature_words:
        prob_count[word] = 1 # 拉普拉斯平滑
    # 遍历每个类别下的新闻
    jieba.enable_parallel(4)
    for news_class in news_classes:
        subfolder = os.path.join(news_folder, news_class)
        news_list = [os.path.join(subfolder, news) for news in os.listdir(subfolder) \
                        if os.path.isfile(os.path.join(subfolder, news))]
        for news in news_list:
            with open(news, 'r') as f:
                content = f.read()
                word_list = jieba.lcut(content)
                for word in prob_count.keys():
                    if word in word_list:
                        prob_count[word] += 1
        news_nums = len(news_list)
        num_of_all_news += news_nums
        prob_classes[news_class] = news_nums
        features_nums = len(feature_words)
        for word in prob_count.keys():
            prob_matrix.loc[word, news_class] = prob_count[word]/(news_nums + features_nums)# 拉普拉斯平滑
    jieba.disable_parallel()
    for cls in prob_classes.keys():
        prob_classes[cls] = prob_classes[cls] / num_of_all_news
    return prob_matrix, prob_classes

def predict_with_content(prob_matrix, prob_classes, feature_words, content):
    word_list = set(jieba.lcut(content))
    result = {}
    for cls in prob_classes.keys():
        result[cls] = np.log(prob_classes[cls])
    for cls in result.keys():
        for word in feature_words:
            if word in word_list:
                result[cls] += np.log(prob_matrix.loc[word, cls])
            else:
                result[cls] += np.log(1 - prob_matrix.loc[word, cls])
    return max(result, key=result.get)

def predict_with_file(news_file, prob_matrix, prob_classes, feature_words):
    with open(news_file) as f:
        news_content = f.read().strip()
    return predict_with_content(prob_matrix, prob_classes, feature_words, content)

def score(news_folder, prob_matrix, prob_classes, feature_words):
    news_classes = [subfolder for subfolder in os.listdir(news_folder) \
                    if os.path.isdir(os.path.join(news_folder, subfolder))]
    y_true = []
    y_predict = []
    for news_class in news_classes:
        subfolder = os.path.join(news_folder, news_class)
        news_list = [os.path.join(subfolder, news) for news in os.listdir(subfolder) \
                        if os.path.isfile(os.path.join(subfolder, news))]
        for news in news_list:
            y_true.append(news_class)
            with open(news, 'r') as f:
                content = f.read()
                y_predict.append(predict_with_content(prob_matrix, prob_classes, feature_words, content))
    return accuracy_score(y_true, y_predict)

if __name__ == "__main__":
    train_folder = 'train_test_data/train'
    test_folder = 'train_test_data/test'
    feature_words = get_feature_words(train_folder)
    prob_matrix, prob_classes = get_probability(train_folder, feature_words)
    acc = score(test_folder, prob_matrix, prob_classes, feature_words)
    print("精确度为:%s" % acc)
    print("测试:%s" % predict_with_file("test.txt"))
