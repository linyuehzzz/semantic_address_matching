import linecache
import random

# query 复制
def copy_query():
    # 取txt文件 的若干行到另一个txt
    with open(r'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/深圳地址数据-20180522-matched-01.txt', 'r', encoding='utf-8') as f1:
        with open(r'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/深圳地址数据-20180522-unmatched-01.txt', 'w', encoding='utf-8')as f2:
            while True:
                line = f1.readline()
                f2.write(line)

# 随机生成不匹配doc
def create_unmatched():
    input_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/深圳地址数据-20180522-matched-02.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/深圳地址数据-20180522-unmatched-02.txt'

    with open(input_file, 'r+', encoding='utf-8') as filehandler:
        linecount = len(filehandler.readlines())

    with open(output_file, 'w', encoding='utf-8') as filehandler2:
        i = 0;
        for i in range(0,linecount):
            i = i + 1;

            k = random.randint(i+6, i+200)
            if (k <= linecount):
                line = linecache.getline(input_file, k)
            else:
                line = linecache.getline(input_file, k - linecount)

            filehandler2.write(line)

# 按列拼接
def append_lines():
    file_1 = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/深圳地址数据-20180522-unmatched-01.txt'
    file_2 = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/深圳地址数据-20180522-unmatched-02.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/深圳地址数据-20180522-unmatched.txt'

    with open(file_1, 'r+', encoding='utf-8') as filehandler:
        linecount = len(filehandler.readlines())

    with open(output_file, 'w', encoding='utf-8') as filehandler2:
        i = 0;
        for i in range(0,linecount):
            i = i + 1;
            line_1 = linecache.getline(file_1, i).strip('\n')
            line_2 = linecache.getline(file_2, i).strip('\n')

            filehandler2.write(line_1 + '\t' + line_2 + '\t' + '0\n')

# 分train/dev/test数据集
def split_dataset(num):

    a = list(range(1, num+1))
    print(len(a),a)
    random.shuffle(a)
    train = a[:int(num * 7 / 10)]
    dev = a[len(train):int(num * 8 / 10)]
    test = a[len(train)+len(dev):]
    print("train",len(train),train)
    print("dev",len(dev),dev)
    print("test",len(test),test)

    input_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/preprocess_data/Address84474.txt'
    output_1 = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/train.txt'
    output_2 = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/dev.txt'
    output_3 = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/test.txt'
    with open(input_file, 'r+', encoding='utf-8') as f:
        line = f.readline()
        with open(output_1, 'w', encoding='utf-8') as o1:
            with open(output_2, 'w', encoding='utf-8') as o2:
                with open(output_3, 'w', encoding='utf-8') as o3:
                    while line:
                        row = line.split()
                        b = row[0]  # 这是选取需要读取的位数
                        if int(b) in train:
                                o1.write(line)
                        if int(b) in dev:
                                o2.write(line)
                        if int(b) in test:
                                o3.write(line)
                        line = f.readline()
                o3.close()
            o2.close()
        o1.close()
    f.close()


split_dataset(84474)