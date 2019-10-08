import logging
import multiprocessing
import os.path
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

# Train word vectors of address elements
def create_model():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    input_dir = '/data/token'
    outp1 = '/model/w2v/word level/GeoW2V.model'
    outp2 = '/model/w2v/word level/word2vec.bin'
    fileNames = os.listdir(input_dir)
    model = Word2Vec(PathLineSentences(input_dir),
                     size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count(), iter=10)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

