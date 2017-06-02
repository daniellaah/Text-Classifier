import os
import sys
import jieba
import numpy as np
from sklearn.externals import joblib

def predict_with_content(classifier, news_content, feature_words):
    word_list = jieba.lcut(news_content)
    x = np.array([1 if word in word_list else 0 for word in feature_words]).reshape(1, -1)
    return classifier.predict(x)[0]

def predict_with_file(classifier, news_file, feature_words):
    with open(news_file) as f:
        news_content = f.read().strip()
    return predict_with_content(classifier, news_content, feature_words)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        news_file = sys.argv[1]
        if not os.path.exists(news_file):
            print("%s 不存在" % news_file)
        elif os.path.exists('朴素贝叶斯新闻分类器_V1/news_clf_model.pkl') and os.path.exists('朴素贝叶斯新闻分类器_V1/news_clf_feature_words.txt'):
            clf = joblib.load('朴素贝叶斯新闻分类器_V1/news_clf_model.pkl')
            feature_words = []
            with open("朴素贝叶斯新闻分类器_V1/news_clf_feature_words.txt", 'r') as f:
                for word in f.readlines():
                    feature_words.append(word.strip())
            print(predict_with_file(clf, news_file, feature_words))
        else:
            print("分类器/特征词 不存在, 请先训练")
    else:
        print("参数错误")
