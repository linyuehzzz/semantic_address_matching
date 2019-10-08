import multiprocessing
import os.path
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

# Train word vectors of address elements (CRF tokenizer)
def create_model():
    input_dir = '\data\CRF'
    outp1 = '\model\w2v_crf\GeoW2V.model'
    outp2 = '\model\w2v_crf\word2vec.bin'
    model = Word2Vec(PathLineSentences(input_dir),
                     size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count(), iter=10)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

create_model()
