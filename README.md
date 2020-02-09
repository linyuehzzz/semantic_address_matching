# A Deep Learning Architecture for Semantic Address Matching


Codes in this repository are for our IJGIS paper [**A Deep Learning Architecture for Semantic Address Matching**](https://www.tandfonline.com/doi/full/10.1080/13658816.2019.1681431).



Data
--------
Data are available at:

  - Shenzhen address corpus (part) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3477633.svg)](https://doi.org/10.5281/zenodo.3477633)
  - Semantic address matching dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3477007.svg)](https://doi.org/10.5281/zenodo.3477007)



Citation
--------
Please cite the following reference if you use the codes 😊.
```
@article{Lin+Kang+Wu+Du+Liu:2020,
  author = {Yue Lin and Mengjun Kang and Yuyang Wu and Qingyun Du and Tao Liu},
  title = {A deep learning architecture for semantic address matching},
  journal = {International Journal of Geographical Information Science},
  volume = {34},
  number = {3},
  pages = {559-576},
  year  = {2020},
  publisher = {Taylor & Francis},
  doi = {10.1080/13658816.2019.1681431}
}
```
Release version of the codes can also be cited as [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3476661.svg)](https://doi.org/10.5281/zenodo.3476661)


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
  - `other_CRF.py` Tokenize using CRF **[Comber, S.; Arribas-Bel, D. (2019) “Machine learning innovations in address matching: A practical comparison of word2vec and CRFs”. Transactions in GIS, 23 (2): 334–348.]** 
  - `other_crf_w2v.py` Train word vectors of address elements (CRF tokenizer)
  - `other_string.py` String similarity-based address matching methods: measure the string relevance
  - `other_w2v_cls.py` Use word2vec embeddings directly for classification: calculat cosine similarity
