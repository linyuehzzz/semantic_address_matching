import tensorflow as tf


class ESIM(object):

    def __init__(self, is_trainning, seq_length, class_num, hidden_num, vocabulary_size, embedding_dim,embedding_matrix,
                 l2_lambda, learning_rate):
        self.is_trainning = is_trainning
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.hidden_num = hidden_num
        self.seq_length = seq_length
        self.class_num = class_num
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate

        # init placeholder
        self.text_a = tf.placeholder(tf.int32, [None, seq_length], name='text_a')
        self.text_b = tf.placeholder(tf.int32, [None, seq_length], name='text_b')
        self.y = tf.placeholder(tf.int32, [None, class_num], name='y')
        # real length
        self.a_length = tf.placeholder(tf.int32, [None], name='a_length')
        self.b_length = tf.placeholder(tf.int32, [None], name='b_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 2. Embedding
        self.vocab_matrix = tf.get_variable(name="vocab_matrix", shape=[vocabulary_size, embedding_dim],
                                            initializer=tf.constant_initializer(embedding_matrix), trainable=True)
        self.text_a_embed = tf.nn.embedding_lookup(self.vocab_matrix, self.text_a)
        self.text_b_embed = tf.nn.embedding_lookup(self.vocab_matrix, self.text_b)

        # 3. Input encoding
        with tf.name_scope('input_encoding'):
            a_bar = self.biLSTMBlock(self.text_a_embed, hidden_num, 'input_encoding/biLSTM', self.a_length)
            b_bar = self.biLSTMBlock(self.text_b_embed, hidden_num, 'input_encoding/biLSTM', self.b_length,
                                     isreuse=True)

        # 4. Local Inference Modeling (attention)
        with tf.name_scope('local_inferenc_modeling'):
            # Locality of inference
            attention_weights = tf.matmul(a_bar, tf.transpose(b_bar, [0, 2, 1]))

            # Local inference collected over sequences
            attentionsoft_a = tf.nn.softmax(attention_weights)
            attentionsoft_b = tf.nn.softmax(tf.transpose(attention_weights))
            attentionsoft_b = tf.transpose(attentionsoft_b)
            a_hat = tf.matmul(attentionsoft_a, b_bar)
            b_hat = tf.matmul(attentionsoft_b, a_bar)

            # Enhancement of local inference information
            a_diff = tf.subtract(a_bar, a_hat)
            a_mul = tf.multiply(a_bar, a_hat)
            b_diff = tf.subtract(b_bar, b_hat)
            b_mul = tf.multiply(b_bar, b_hat)
            self.m_a = tf.concat([a_bar, a_hat, a_diff, a_mul], axis=2)
            self.m_b = tf.concat([b_bar, b_hat, b_diff, b_mul], axis=2)

        # 5. Inference Composition
        with tf.name_scope("inference_composition"):
            # The composition layer
            v_a = self.biLSTMBlock(self.m_a, hidden_num, 'inference_composition/biLSTM', self.a_length)
            v_b = self.biLSTMBlock(self.m_b, hidden_num, 'inference_composition/biLSTM', self.b_length, isreuse=True)

            # Average and max pooling
            v_a_avg = tf.reduce_mean(v_a, axis=1)
            v_b_avg = tf.reduce_mean(v_b, axis=1)
            v_a_max = tf.reduce_max(v_a, axis=1)
            v_b_max = tf.reduce_max(v_b, axis=1)
            v = tf.concat([v_a_avg, v_a_max, v_b_avg, v_b_max], axis=1)

        # 6. Output(MLP/ relu & tahn/ softmax)
        with tf.name_scope("output"):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(v, self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs, hidden_num, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                self.logits = tf.layers.dense(outputs, class_num, tf.nn.tanh, kernel_initializer=initializer)
            self.score = tf.nn.softmax(self.logits, name='score')
            self.prediction = tf.argmax(self.score, 1, name="prediction")

        # cost: multi-class loss entropy loss
        with tf.name_scope('cost'):
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            self.cost = tf.reduce_mean(self.cost)
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            self.loss = l2_loss + self.cost

        # calculate classification accuracy
        self.accuracy = tf.reduce_mean(
             tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32))

        if not is_trainning:
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)

        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def biLSTMBlock(self, inputs, num_units, scope, seq_len=None, isreuse=False):
        with tf.variable_scope(scope, reuse=isreuse):
            # LSTM (F & B)
            # num_units: output dimension
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)

            # Define Bi-RNN
            (a_outputs, a_output_states) = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell,
                                                                           inputs,
                                                                           sequence_length=seq_len,
                                                                           dtype=tf.float32)

            # Concatenate F & B LSTM
            a_bar = tf.concat(a_outputs, axis=2)
            return a_bar


if __name__ == '__main__':
    esim = ESIM(True, 20, 2, 10000, 300, 300, 0.001, 0.0001)
