from pyhanlp import *
import os


def CRF():
    file_nums = 0
    count = 0
    url_1 = '/data/corpus/'
    url_2 = '/data/CRF/'
    filenames = os.listdir(url_1)

    HanLP.Config.ShowTermNature = False
    CRFnewSegment = HanLP.newSegment("crf")
    for file in filenames:
        print('Starting ' + str(file_nums) + 'file word Segmentation!')
        segment_file = open(url_2 + file + '_crf', 'a', encoding='utf8')
        with open(url_1 + file, encoding='utf8') as f:
            text = f.readlines()
            for sentence in text:
                print(sentence)
                sentence = CRFnewSegment.seg(sentence)
                for i in range(len(sentence)):
                    segment_file.write(str(sentence[i]) + " ")
            del text
            f.close()
        segment_file.close()
        print('Finished ' + str(file_nums) + 'file word Segmentation!')
        file_nums += 1


CRF()
