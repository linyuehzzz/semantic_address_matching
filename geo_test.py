import tensorflow as tf
import numpy as np
import geo_data_prepare
from keras.preprocessing.sequence import pad_sequences
import geo_config as config
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn import metrics

data_pre = geo_data_prepare.Data_Prepare()
con = config.Config()

class Infer(object):
    def __init__(self):
        self.checkpoint_file = tf.train.latest_checkpoint('D:/Lydia/PycharmProjects/Deep learning for geocoding/model/esim')
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.text_a = graph.get_operation_by_name("esim_model/text_a").outputs[0]
                self.text_b = graph.get_operation_by_name("esim_model/text_b").outputs[0]
                self.a_length = graph.get_operation_by_name("esim_model/a_length").outputs[0]
                self.b_length = graph.get_operation_by_name("esim_model/b_length").outputs[0]
                self.drop_keep_prob = graph.get_operation_by_name("esim_model/dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.prediction = graph.get_operation_by_name("esim_model/output/prediction").outputs[0]
                self.score = graph.get_operation_by_name("esim_model/output/score").outputs[0]
                self.loss = graph.get_operation_by_name("esim_model/output/score").outputs[0]

    def get_length(self, text):
        # sentence length
        lengths = []
        for sample in text:
            count = 0
            for index in sample:
                if index != 0:
                    count += 1
                else:
                    break
            lengths.append(count)
        return lengths

    def to_categorical(self, y, nb_classes=None):
        y = np.asarray(y, dtype='int32')
        if not nb_classes:
            nb_classes = np.max(y) + 1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y

    def get_batches(self, texta, textb, tag):
        num_batch = int(len(texta) / con.Batch_Size)
        for i in range(num_batch):
            a = texta[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            b = textb[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            t = tag[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            yield a, b, t

    def infer(self):
        # transfer to vector
        test_texta_index = data_pre.readfile('/data/dataset/test_code_a.txt')
        test_texta_index = pad_sequences(test_texta_index, con.maxLen, padding='post')
        print(test_texta_index[0])
        print(len(test_texta_index))
        test_textb_index = data_pre.readfile('/data/dataset/test_code_b.txt')
        test_textb_index = pad_sequences(test_textb_index, con.maxLen, padding='post')
        print(test_textb_index[0])
        print(len(test_textb_index))
        test_tag = data_pre.readfile('/data/dataset/test_lable.txt')
        test_tag = self.to_categorical(np.asarray(test_tag, dtype='int32'))
        print(test_tag[0])
        print(len(test_tag))

        # shuffle dev
        index = [x for x in range(len(test_texta_index))]
        random.shuffle(index)
        test_texta_new = [test_texta_index[x] for x in index]
        test_textb_new = [test_textb_index[x] for x in index]
        test_tag_new = [test_tag[x] for x in index]

        y_pred = []
        y_true = []
        for texta, textb, tag in tqdm(
                self.get_batches(test_texta_new, test_textb_new, test_tag_new)):
            feed_dict = {
                self.text_a: texta,
                self.text_b: textb,
                self.drop_keep_prob: 1.0,
                self.a_length: np.array(self.get_length(texta)),
                self.b_length: np.array(self.get_length(textb))
            }
            y, s = self.sess.run([self.prediction, self.score], feed_dict)
            y_pred.extend(y)
            y_tag = [np.nonzero(x)[0][0] for x in tag]
            y_true.extend(y_tag)

        f1 = f1_score(np.array(y_true), np.array(y_pred), average='weighted')
        print('Outputs:\n', metrics.classification_report(np.array(y_true), y_pred))


infer = Infer()
infer.infer()
