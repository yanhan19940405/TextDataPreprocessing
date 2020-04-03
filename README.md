# TextDataPreprocessing
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

