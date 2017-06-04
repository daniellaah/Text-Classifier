## 文本分类
### 运行环境
- python3.5

### 新闻数据获取
数据源: [搜狗实验室](http://www.sogou.com/labs/resource/cs.php)

- `news_sohusite_xml.dat`
- `categories_2012.txt`

说明: 第二个文件包含url与新闻类别的映射关系.  

### 00-数据预处理
对如下几个类别进行实验:

|  类别  |            URL            |
| :--: | :-----------------------: |
|   IT  | http://it.souhu.com/   |
|  财经  | http://business.sohu.com/ |
|  体育  |  http://sports.sohu.com/  |
|  娱乐  |   http://yule.sohu.com/   |


```
cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep -E "<content>|<url>" > sohu_news.txt
```    

```
python 00-数据预处理/news_data_processing.py
```
### 01-sklearn Naive Bayes新闻分类器
#### 依赖: [jieba](https://github.com/fxsjy/jieba), [sklearn](http://scikit-learn.org/)

- Bernoulli Naive Bayes
```
python 01-sklearn朴素贝叶斯新闻分类器/Bernoulli_NaiveBayes.py 01-sklearn朴素贝叶斯新闻分类器
```
![](https://github.com/daniellaah/Text-Classifier/blob/master/01-sklearn%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%96%B0%E9%97%BB%E5%88%86%E7%B1%BB%E5%99%A8/screenshot_1138.png)

-  Multinomial Naive Bayes
```
python 01-sklearn朴素贝叶斯新闻分类器/Multinomial_NaiveBayes.py 01-sklearn朴素贝叶斯新闻分类器
```
![](https://github.com/daniellaah/Text-Classifier/blob/master/01-sklearn%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%96%B0%E9%97%BB%E5%88%86%E7%B1%BB%E5%99%A8/screenshot_1139.png)

### 02-手动实现Bernoulli Naive Bayes新闻分类器
```
python 02-手动实现Bernoulli_NaiveBayes新闻分类器/Bernoulli_NaiveBayes.py
```
![](https://github.com/daniellaah/Text-Classifier/blob/master/02-手动实现Bernoulli_NaiveBayes新闻分类器/screenshot_1147.png)

### 03-手动实现Multinomial Naive Bayes新闻分类器
```
python 03-手动实现Multinomial_NaiveBayes新闻分类器/Multinomial_NaiveBayes.py
```
![](https://github.com/daniellaah/Text-Classifier/blob/master/03-手动实现Multinomial_NaiveBayes新闻分类器/screenshot_1148.png)

### 04-手动实现Softmax Regression新闻分类器
```
python 04-手动实现SoftmaxRegression新闻分类器/SoftmaxRegression.py
```
![](https://github.com/daniellaah/Text-Classifier/blob/master/04-手动实现SoftmaxRegression新闻分类器/screenshot_1145.png)

### 05-sklearn三种模型对比
```
python 05-sklearn三种模型对比/News_Classifier.py
```
![](https://github.com/daniellaah/Text-Classifier/blob/master/05-sklearn三种模型对比/screenshot_1146.png)

