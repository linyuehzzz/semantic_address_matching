class Config(object):

    def __init__(self):
        self.maxLen = 100 # length of padding
        self.hidden_num = 50  # number of hidden nodes
        self.l2_lambda = 0.01  # L2 regularization parameter
        self.learning_rate = 0.0001  # learning rate
        self.dropout_keep_prob = 0.5  # dropout probability
        self.attn_size = 200  # attention size
        self.K = 2

        self.epoch = 20  # epoch
        self.Batch_Size = 75  # mini-batch size
