import numpy as np
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
import re
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
class TextData:#text size:(n,len),type:list(text)
    def __init__(self,tokenmode,vocabsize):
        self.tokenmode=tokenmode
        self.vocabsize=vocabsize
    def strlen_count(self,text):#统计序列长度
        str=[ len(i) for i in text]
        print(str)
        strlen=sorted(str)
        print(strlen)
        return strlen[int(len(str)*0.9)],strlen
    def strlen_canvas(self,text):#序列长度可视化
        _,b=self.strlen_count(text)
        plt.ylabel("strlen")
        plt.xlabel("str num")
        plt.title('String Length')
        ticks = [int(i) for i in range(len(b))]
        plt.xticks(range(len(b)), ticks)
        y=b
        x=[int(i) for i in range(len(b))]
        plt.bar(x, y, facecolor='red', width=0.2)
        for x, y in zip(x, y):
            plt.text(x + 0.05, y + 0.05, '%i' % y, ha='right', va='bottom')
        plt.legend()
        plt.show()
        plt.close()
    def text_token(self,text):#序列分词
        token=[]
        if self.tokenmode=="Word":
            for i in text:
                token.append(jieba.lcut(i))
        elif self.tokenmode=='Char':
            for i in text:
                token.append(list(i))
        return token

    def text_dict(self,text):#生成索引词典
        vocab={}
        vocab_list=[]
        token=self.text_token(text)
        for a in token:  # 生成词典
            for b in a:
                vocab_list.append(b)
        word_counts = collections.Counter(vocab_list)
        word_counts = word_counts.most_common(self.vocabsize)
        for index, con in enumerate(word_counts):
            vocab[str(con[0])] = index + 2
        vocab["UNK"] = 1
        vocab["PAD"] = 0
        return vocab
    def dict_save(self,filename,text):#词典保存代码
        vocab=self.text_dict(text)
        output = open(filename, 'wb')
        pickle.dump(vocab, output)
        output.close()
    def text_str(self,text):#文本补长代码
        data_text=self.text_token(text)
        MAXLEN,_=self.strlen_count(text)
        for j in range(len(data_text)):  # 文本补长
            if len(data_text[j]) <= MAXLEN:
                data_text[j].extend(['PAD'] * (MAXLEN - len(data_text[j])))
            elif len(data_text[j]) > MAXLEN:
                data_text[j] = data_text[j][0:MAXLEN]
        return data_text

    def text_matrix(self,text):#文本矩阵化
        data_text=self.text_str(text)
        vocab=self.text_dict(text)
        for m in range(len(data_text)):
            for n in range(len(data_text[m])):
                if data_text[m][n] in vocab.keys():
                    data_text[m][n] = vocab[data_text[m][n]]
                elif data_text[m][n] not in vocab.keys():
                    data_text[m][n] = vocab['UNK']
        return np.array(data_text)
    def text_tf_idf(self,text):#生成TFIDF矩阵
        token=self.text_token(text)
        token=[" ".join(j) for j in token]
        print(token)
        tf_idf_value = TfidfVectorizer()
        tf_idf_matrix = tf_idf_value.fit_transform(token)
        print(tf_idf_value.vocabulary_)
        return  tf_idf_matrix.toarray()

if __name__ == '__main__':

    s=['据美国NBC新闻4月2日报道，美国佛罗里达州附近水域上，荷美邮轮公司的“赞丹”号上已有4人死亡，其中至少2人死于新冠病毒感染，其他9人检测呈阳性，另有179人出现了类似流感症状。','NBC报道称，目前，“赞丹”号及前去支援的“鹿特丹”号获准停靠佛罗里达州一港口，9名确诊患者将转移到当地医院，另外45名症状轻微的患病乘客不适合出行，将继续留在船上。',
       '而微博网友“暴以素”就在“赞丹”号上，在出现类似流感症状且病情加重后“失联”数日，引得一些网友担忧。']

    datadeal=TextData(tokenmode="Word",vocabsize=200)
    a,b=datadeal.strlen_count(s)

    print(a,b)
    datadeal.strlen_canvas(s)

    token=datadeal.text_token(s)
    print(token)

    vocab=datadeal.text_dict(s)
    print(vocab)

    data=datadeal.text_str(s)
    datanum=datadeal.text_matrix(s)

    datatfidf=datadeal.text_tf_idf(s)
    print(1)
