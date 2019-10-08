import jieba
import random
from tensorflow.contrib import learn
import numpy as np

UNKNOWN = '<UNK>'
PADDING = '<PAD>'

class Data_Prepare(object):

    # 读入文件（用词典序号表示单词）
    def readfile(self, filename):
        text_index = []
        tag = []
        with open(filename, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                text_index.append([int(i) for i in line])
        return text_index

    # 获取停用词
    def get_stopwords(self):
        stopwords = []
        with open("D:/Lydia/PycharmProjects/Deep learning for geocoding/data/Stopwords.txt", "r",
                  encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                stopwords.append(line.strip())
        return stopwords

    # 分词
    def pre_processing(self, text):
        # 删除（）里的内容
        stopwords = self.get_stopwords()
        words = jieba.cut(text)
        words_seg = []
        for word in words:
            if word not in stopwords:
                words_seg.append(word)
        return words_seg

    # 文字转换为词典中的序号
    def sentence2Index(self, text, vocabDict):
        sList = []
        for word in text:
            if word in vocabDict:
                sList.append(vocabDict.index(word))
            elif word not in vocabDict:
                sList.append(0)
        return sList

if __name__ == '__main__':
    data_pre = Data_Prepare()