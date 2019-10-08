import logging
import multiprocessing
import os.path
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
input_dir = 'C:/Users/linyue/PycharmProjects/Deep learning for geocoding/data/corpus/'
output_dir = 'C:/Users/linyue/PycharmProjects/Deep learning for geocoding/data/token_char/'


# 按字符分词
def add_blank(input_dir):
    for filename in os.listdir(input_dir):
        pathname = os.path.join(input_dir, filename)
        with open(pathname, 'r+', encoding='utf-8') as filehandler:
            with open(os.path.join(output_dir, filename), 'w',encoding='utf-8') as filehandler2:
                filehandler2.write(''.join([f + ' ' for fh in filehandler for f in fh]))

add_blank(input_dir)


# 生成字向量
def create_model():
    outp1 = 'C:/Users/linyue/PycharmProjects/Deep learning for geocoding/model/w2v/char level/GeoW2V.model'
    outp2 = 'C:/Users/linyue/PycharmProjects/Deep learning for geocoding/model/w2v/char level/word2vec.bin'
    fileNames = os.listdir(output_dir)
    # 训练模型 输入语料目录 embedding size 256,共现窗口大小10,去除出现次数5以下的词,多线程运行,迭代10次
    model = Word2Vec(PathLineSentences(output_dir),
                     size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count(), iter=10)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)