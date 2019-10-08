class Config(object):

    def __init__(self):
        self.maxLen = 100 # padding长度
        self.hidden_num = 50  # 隐藏层规模
        self.l2_lambda = 0.01  # 正则化参数
        self.learning_rate = 0.0001
        self.dropout_keep_prob = 0.5
        self.attn_size = 200
        self.K = 2

        self.epoch = 20
        self.Batch_Size = 75