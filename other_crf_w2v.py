import multiprocessing
import os.path
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

# 生成词向量
def create_model():
    input_dir = 'D:\Lydia\PycharmProjects\Deep learning for geocoding\data\CRF'
    outp1 = 'D:\Lydia\PycharmProjects\Deep learning for geocoding\model\w2v_crf\GeoW2V.model'
    outp2 = 'D:\Lydia\PycharmProjects\Deep learning for geocoding\model\w2v_crf\word2vec.bin'
    # 训练模型 输入语料目录 embedding size 256,共现窗口大小10,去除出现次数5以下的词,多线程运行,迭代10次
    model = Word2Vec(PathLineSentences(input_dir),
                     size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count(), iter=10)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

create_model()