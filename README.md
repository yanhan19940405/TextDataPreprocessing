# TextDataPreprocessing
##注意##
由于github某些组件最近被墙，导致图片无法正常加载，请按照如下方式处理：

1、更改hosts文件，添加如下信息，刷新网页即可解决此问题：

>> ```
>> 52.74.223.119 github.com
>> 192.30.253.119 gist.github.com
>> 54.169.195.247 api.github.com
>> 185.199.111.153 assets-cdn.github.com
>> 151.101.76.133 raw.githubusercontent.com
>> 151.101.108.133 user-images.githubusercontent.com
>> 151.101.76.133 gist.githubusercontent.com
>> 151.101.76.133 cloud.githubusercontent.com
>> 151.101.76.133 camo.githubusercontent.com
>> 52.74.223.119 github.com
>> 192.30.253.119 gist.github.com
>> 54.169.195.247 api.github.com
>> 185.199.111.153 assets-cdn.github.com
>> 151.101.76.133 raw.githubusercontent.com
>> 151.101.108.133 user-images.githubusercontent.com
>> 151.101.76.133 gist.githubusercontent.com
>> 151.101.76.133 cloud.githubusercontent.com
>> 151.101.76.133 camo.githubusercontent.com
>> ```

2、可以将源码图片文件夹中找到相应图片。

文本数据预处理小工具，支持一行代码将文本序列转换为相应数值矩阵和TFIDF数值矩阵，便于后续直接进行模型实验

本工具所需环境如下：

>> ```
>> import numpy as np
>> import matplotlib.pyplot as plt
>> import jieba
>> import jieba.analyse
>> import re
>> import pandas as pd
>> import collections
>> from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
>> ```

本工具输入参数有两个，是：tokenmode、vocabsize，其中tokemode表示分词方式，取值为"Char"或者"Word",为string类型；vocabsize表示词表大小，为int类型
该工具输入文本集合s类型为二维列表，形状为（n,len）,其中n表示句子个数，len表示每一条句子长度

若输出数值矩阵，输出数据类型为numpy矩阵类型，形状为(n,MAXLEN)，MAXLEN表示句子序列最大补长长度

若输出文TFIDF特征矩阵，则形状为（n,d）,d表示词表大小，此处与vocabsize数值无关。

对于新人而言，本工具可以直接使用text_matrix（）方法将输入文本序列s转换为数值矩阵。

运行效果如下：

![图1 文本数值矩阵](https://github.com/yanhan19940405/TextDataPreprocessing/blob/master/img/2.png?raw=true)

本工具也可以使用text_tf_idf()方法，一行代码生成TFIDF矩阵，运行效果如下：

![图2 文本数值矩阵](https://github.com/yanhan19940405/TextDataPreprocessing/blob/master/img/3.png?raw=true)

最后，本工具也对支持文本长度可视化，运行效果如下所示：

![图3 文本数值矩阵](https://github.com/yanhan19940405/TextDataPreprocessing/blob/master/img/1.png?raw=true)

除此外本工具也支持生成索引词典与其他功能，若需扩展可以直接下载代码到本地更改，欢迎试用

