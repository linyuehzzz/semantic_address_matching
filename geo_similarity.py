import distance
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Levenshtein distance
def edit_distance(s1, s2):
    return distance.levenshtein(s1, s2)

def Levenshtein_total():
    d_sum = 0
    count = 0
    filename = '/data/preprocess_data/Address84474.txt'
    output = '/data/preprocess_data/Levenshtein_total.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                d = edit_distance(s1, s2)
                o.write(s1 + '\t' + s2 + '\t' + str(d) + '\n')
                d_sum = d_sum + d
                count = count + 1
            d_total_mean = d_sum/count
            print('Levenshtein distance', d_total_mean)

def Levenshtein_matched():
    d_sum = 0
    count = 0
    filename = '/data/preprocess_data/Shenzhen_address_data-20180522-matched.txt'
    output = '/data/preprocess_data/Levenshtein_matched.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                d = edit_distance(s1, s2)
                o.write(s1 + '\t' + s2 + '\t' + str(d) + '\n')
                d_sum = d_sum + d
                count = count + 1
            d_matched_mean = d_sum/count
            print('Levenshtein distance', d_matched_mean)

def Levenshtein_unmatched():
    d_sum = 0
    count = 0
    filename = '/data/preprocess_data/Shenzhen_address_data-20180522-unmatched.txt'
    output = '/data/preprocess_data/Levenshtein_unmatched.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                d = edit_distance(s1, s2)
                o.write(s1 + '\t' + s2 + '\t' + str(d) + '\n')
                d_sum = d_sum + d
                count = count + 1
            d_unmatched_mean = d_sum/count
            print('Levenshtein distance', d_unmatched_mean)

# Jaccard相似度
def Jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator

def Jaccard_total():
    s_sum = 0
    count = 0
    filename = '/data/preprocess_data/Address84474.txt'
    output = '/data/preprocess_data/Jaccard_total.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                s = Jaccard_similarity(s1, s2)
                o.write(s1 + '\t' + s2 + '\t' + str(s) + '\n')
                s_sum = s_sum + s
                count = count + 1
            s_total_mean = s_sum/count
            print('Jaccard similarity', s_total_mean)

def Jaccard_matched():
    s_sum = 0
    count = 0
    filename = '/data/preprocess_data/Shenzhen_address_data-20180522-matched.txt'
    output = '/data/preprocess_data/Jaccard_matched.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                s = Jaccard_similarity(s1, s2)
                o.write(s1 + '\t' + s2 + '\t' + str(s) + '\n')
                s_sum = s_sum + s
                count = count + 1
            s_matched_mean = s_sum/count
            print('Jaccard similarity', s_matched_mean)

def Jaccard_unmatched():
    s_sum = 0
    count = 0
    filename = '/data/preprocess_data/Shenzhen_address_data-20180522-unmatched.txt'
    output = '/data/preprocess_data/Jaccard_unmatched.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                s = Jaccard_similarity(s1, s2)
                o.write(s1 + '\t' + s2 + '\t' + str(s) + '\n')
                s_sum = s_sum + s
                count = count + 1
            s_unmatched_mean = s_sum/count
            print('Jaccard similarity', s_unmatched_mean)

# 长度差
def diff_total():
    d_sum = 0
    count = 0
    filename = '/data/preprocess_data/Address84474.txt'
    output = '/data/preprocess_data/diff_total.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                d = abs(len(s1) - len(s2))
                o.write(s1 + '\t' + s2 + '\t' + str(d) + '\n')
                d_sum = d_sum + d
                count = count + 1
            d_total_mean = d_sum/count
            print('length difference', d_total_mean)

def diff_matched():
    d_sum = 0
    count = 0
    filename = '/data/preprocess_data/Shenzhen_address_data-20180522-matched.txt'
    output = '/data/preprocess_data/diff_matched.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                d = abs(len(s1) - len(s2))
                o.write(s1 + '\t' + s2 + '\t' + str(d) + '\n')
                d_sum = d_sum + d
                count = count + 1
            d_matched_mean = d_sum/count
            print('length difference', d_matched_mean)

def diff_unmatched():
    d_sum = 0
    count = 0
    filename = '/data/preprocess_data/Shenzhen_address_data-20180522-unmatched.txt'
    output = '/data/preprocess_data/diff_unmatched.txt'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                d = abs(len(s1) - len(s2))
                o.write(s1 + '\t' + s2 + '\t' + str(d) + '\n')
                d_sum = d_sum + d
                count = count + 1
            d_unmatched_mean = d_sum/count
            print('length difference', d_unmatched_mean)


diff_unmatched()
