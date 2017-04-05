## 基于朴素贝叶斯的新闻分类器
### 新闻数据获取
搜狗实验室: http://www.sogou.com/labs/resource/cs.php
我们还需要下载类别标记说明(URL到类别的映射关系)`categories_2012.txt`, 从这个文件中我们可以得到不同的url对应的类别.  
例如`<url>http://gongyi.sohu.com/20120706/n347457739.shtml</url>`, 通过查看映射关系:我们知道该新闻属于公益类.
### 数据预处理
我们仅对如下数据比较多的几个类别进行实验:
|  类别  |            URL            |
| :--: | :-----------------------: |
|  财经  | http://business.sohu.com/ |
|  体育  |  http://sports.sohu.com/  |
|  娱乐  |   http://yule.sohu.com/   |
`cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep -E "<content>|<url>" > sohu_news.txt`  
`python news_data_processing.py`

### 模型训练与存储
`python model_training_save.py`

### 模型加载与预测
