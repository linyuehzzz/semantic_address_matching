import tensorflow as tf
import numpy as np
import geo_data_prepare
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import geo_config as config
from sklearn.metrics import pairwise

con = config.Config()
data_pre = geo_data_prepare.Data_Prepare()
maxLen = 15
np.set_printoptions(threshold=np.inf)

class w2v_sim(object):

    def get_sim(self, seq_length, vocabulary_size, embedding_dim, embedding_matrix):
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix

        # 1. Init placeholder
        self.text_a = tf.placeholder(tf.int32, [None, self.seq_length], name='text_a')
        self.text_b = tf.placeholder(tf.int32, [None, self.seq_length], name='text_b')

        # 2. Embedding
        self.vocab_matrix = tf.get_variable(name='vocab_matrix', shape=[self.vocabulary_size, self.embedding_dim],
                                            initializer=tf.constant_initializer(self.embedding_matrix), trainable=True)
        self.text_a_embed = tf.nn.embedding_lookup(self.vocab_matrix, self.text_a)
        self.text_b_embed = tf.nn.embedding_lookup(self.vocab_matrix, self.text_b)

        # 3. Similarity
        self.a_val = tf.sqrt(tf.reduce_sum(tf.matmul(self.text_a_embed, self.text_a_embed, transpose_b=True), axis=1))
        self.b_val = tf.sqrt(tf.reduce_sum(tf.matmul(self.text_b_embed, self.text_b_embed, transpose_b=True), axis=1))
        self.denom = tf.multiply(self.a_val, self.b_val)
        self.num = tf.reduce_sum(tf.matmul(self.text_a_embed, self.text_b_embed, transpose_b=True), axis=1)
        self.similarity = tf.div(self.num, self.denom)

    def get_batches(self, texta, textb, tag):
        num_batch = int(len(texta))
        for i in range(num_batch):
            a = texta[i:(i + 1)]
            b = textb[i:(i + 1)]
            t = tag[i:(i + 1)]
            yield a, b, t

    def run(self):
        # Load word embeddings
        vocab = []
        embed = []
        cnt = 0
        fr = open(r'/model/w2v_crf/word2vec.bin', 'r', encoding='UTF-8')
        line = fr.readline().strip()
        word_dim = int(line.split(' ')[1])
        vocab.append('unk')
        embed.append([0] * word_dim)
        for line in fr:
            row = line.strip().split(' ')
            vocab.append(row[0])
            embed.append(row[1:])
        print('Loaded word2vec!')
        fr.close()
        print(vocab[0], vocab[1], vocab[2])
        print(embed[0])
        print(embed[1])
        vocab_size = len(vocab)
        print(vocab_size)
        embedding_dim = len(embed[0])
        print(embedding_dim)
        embedding_matrix = np.asarray(embed)

        # Read test dataset
        texta_index = data_pre.readfile(
            '/data/dataset/test_code_a.txt')
        texta_index = pad_sequences(texta_index, maxLen, padding='post')
        print(texta_index[0])
        print(len(texta_index))
        textb_index = data_pre.readfile(
            'g/data/dataset/test_code_b.txt')
        textb_index = pad_sequences(textb_index, maxLen, padding='post')
        print(textb_index[0])
        print(len(textb_index))
        tag = data_pre.readfile('/data/dataset/test_lable.txt')

        # Convert text to vector
        y_true = []
        sim = []
        model = self.get_sim(len(texta_index[0]), vocab_size, embedding_dim, embedding_matrix)
        with open(r'/data/other/crf_w2v_test.csv', 'w',
                  encoding='utf8') as f:
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for texta, textb, tag in tqdm(self.get_batches(texta_index, textb_index, tag)):
                    feed_dict = {
                        self.text_a: texta,
                        self.text_b: textb,
                    }
                    s = sess.run([self.similarity], feed_dict)
                    y_true.append(tag)
                    sim.append(s)

                    tag_new = tag[0]
                    f.write(str(tag_new[0]) + ',')
                    s_new = s[0]
                    s_new_2 = s_new[0]
                    for i in range(maxLen):
                        f.write(str(s_new_2[i]) + ',')
                    f.write('\n')

sim = w2v_sim()
sim.run()
