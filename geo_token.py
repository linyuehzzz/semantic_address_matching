import jieba
import os
import logging

# Load custom dictionaries
jieba.load_userdict('/data/GeoDicv2.txt')


# Load stopwords
def get_stopwords():
    stopwords = []
    with open("/data/Stopwords.txt", "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


stopwords = get_stopwords()


# Tokenize
def segment():
    file_nums = 0
    count = 0
    url_1 = '/data/corpus/'
    url_2 = '/data/token/'
    filenames = os.listdir(url_1)
    for file in filenames:
        logging.info('Starting ' + str(file_nums) + 'file word Segmentation!')
        segment_file = open(url_2 + file + '_segment', 'a', encoding='utf8')
        with open(url_1 + file, encoding='utf8') as f:
            text = f.readlines()
            for sentence in text:
                sentence = list(jieba.cut(sentence))
                sentence_segment = []
                for word in sentence:
                    if word not in stopwords:
                        sentence_segment.append(word)
                segment_file.write(" ".join(sentence_segment))
            del text
            f.close()
        segment_file.close()
        logging.info('Finished ' + str(file_nums) + 'file word Segmentation!')
        file_nums += 1


segment()
