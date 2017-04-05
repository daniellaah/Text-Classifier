# 基于朴素贝叶斯的新闻分类器
## 新闻数据获取
搜狗实验室: http://www.sogou.com/labs/resource/cs.php
下载并解压后得到news_sohusite_xml.dat文件, 由于该文件是gbk编码, 使用`iconv`命令进行编码格式转换, 使用`grep -E`截取包含`<content>`和`<url>`的行的内容, 将输出重定向到news.txt中:   
```cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 | grep -E "<content>|<url>" > sohu_news.txt```
但在我的机器上老是报错`iconv: (stdin):5:291: cannot convert`, 在`iconv`命令最后加一个`-c`表示丢弃那些无法被转换的字符:
```cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep -E "<content>|<url>" > sohu_news.txt```
我们还需要下载类别标记说明(URL到类别的映射关系)`categories_2012.txt`, 从这个文件中我们可以得到不同的url对应的类别.  
例如`<url>http://gongyi.sohu.com/20120706/n347457739.shtml</url>`, 通过查看映射关系:我们知道该新闻属于公益类的.
我们仅对如下数据比较多的几个类别进行实验:

|类别|URL|
|:-:|:-:|
|财经|http://business.sohu.com/ |
|体育 |http://sports.sohu.com/| |
|娱乐 |http://yule.sohu.com/ |
