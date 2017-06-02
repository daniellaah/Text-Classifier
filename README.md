## 文本分类
### 运行环境
- python3.5

### 新闻数据获取
数据源: [搜狗实验室](http://www.sogou.com/labs/resource/cs.php)

- `news_sohusite_xml.dat`
- `categories_2012.txt`

说明: 第二个文件包含url与新闻类别的映射关系.  
例如`<url>http://gongyi.sohu.com/20120706/n347457739.shtml</url>`, 通过查看映射关系:我们知道该新闻属于公益类.

### 00-数据预处理
对如下几个类别进行实验:

|  类别  |            URL            |
| :--: | :-----------------------: |
|  财经  | http://business.sohu.com/ |
|  体育  |  http://sports.sohu.com/  |
|  娱乐  |   http://yule.sohu.com/   |
|   IT   | http://it.souhu.com/   |

```
cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep -E "<content>|<url>" > sohu_news.txt
```    

```
python 00-数据预处理/news_data_processing.py
```
### 01-sklearn朴素贝叶斯新闻分类器
#### 依赖: [jieba](https://github.com/fxsjy/jieba), [sklearn](http://scikit-learn.org/)

- 伯努利朴素贝叶斯
```
python 01-sklearn朴素贝叶斯新闻分类器/Bernoulli_NaiveBayes.py
```
- 多项式朴素贝叶斯
```
python 01-sklearn朴素贝叶斯新闻分类器/Multinomial_NaiveBayes.py
```
### 02-手动实现朴素贝叶斯新闻分类器
- 手动实现伯努利朴素贝叶斯
```
python 02-手动实现朴素贝叶斯新闻分类器/Bernoulli_NaiveBayes.py
```
- 手动实现多项式朴素贝叶斯
```
python 02-手动实现朴素贝叶斯新闻分类器/Multinomial_NaiveBayes.py
```
### 03-三种模型对比(NB,LR,SVM)
```
python 03-三种模型对比(NB,LR,SVM)/词袋模型/News_Classifier.py
```