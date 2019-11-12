# A Deep Learning Architecture for Semantic Address Matching


Codes in this repository are for our paper **A deep learning architecture for semantic address matching**


Citations [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3476673.svg)](https://doi.org/10.5281/zenodo.3476673)
--------
```
@article{doi:10.1080/13658816.2019.1681431,
author = {Yue Lin and Mengjun Kang and Yuyang Wu and Qingyun Du and Tao Liu},
title = {A deep learning architecture for semantic address matching},
journal = {International Journal of Geographical Information Science},
volume = {0},
number = {0},
pages = {1-18},
year  = {2019},
publisher = {Taylor & Francis},
doi = {10.1080/13658816.2019.1681431},
}
```



Data
--------
Data are available at:

  - Shenzhen address corpus (part) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3477633.svg)](https://doi.org/10.5281/zenodo.3477633)
  - Semantic address matching dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3477007.svg)](https://doi.org/10.5281/zenodo.3477007)



Details
--------
Below is an overview of each file in this repository.

  - `geo_config.py` Hyperparameter settings for the ESIM
  - `geo_data_prepare.py` Tokenize the corpus and convert each address element into index
  - `geo_data_processor.py` Process the labeled address dataset and divide it into training, development and test sets
  - `geo_ESIM.py` Implementation of the enhanced sequential inference model (ESIM)
  - `geo_similarity.py` Calculate statistical characteristics of the labeled address dataset
  - `geo_test.py` Output predictive results of the ESIM on the test set
  - `geo_token.py` Tokenize with the Jieba library
  - `geo_train.py` Train the ESIM and evaluate its accuracy on the development set
  - `geo_word2vec.py` Train word vectors of address elements
  - `other_CRF.py` Tokenize using CRF **[Comber and Arribas-Bel (2019)]** 
  - `other_crf_w2v.py` Train word vectors of address elements (CRF tokenizer)
  - `other_string.py` String similarity-based address matching methods: measure the string relevance
  - `other_w2v_cls.py` Use word2vec embeddings directly for classification: calculat cosine similarity
