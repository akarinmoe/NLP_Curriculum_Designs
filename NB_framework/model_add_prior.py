# -*- coding: utf-8 -*-
import jieba
import numpy as np

train_path = "./train.txt"
test_path = "./test.txt"

sum_words_neg = 0   # 训练集负向语料的总词数（用于计算词频）
sum_words_pos = 0   # 训练集正向语料的总词数

neg_sents_train = []  # 训练集中负向句子
pos_sents_train = []  # 训练集中正向句子
neg_sents_test = []  # 测试集中负向句子
pos_sents_test = []  # 测试集中正向句子
stopwords = []  # 停用词

def mystrip(ls):
    for i in range(len(ls)):
        ls[i] = ls[i].strip("\n")
    return ls

def remove_stopwords(_words):
    _i = 0
    for _ in range(len(_words)):
        if _words[_i] in stopwords:
            _words.pop(_i)
        else:
            _i += 1
    return _words

def my_init():
    neg_words = []  # 负向词列表
    _neg_dict = {}  # 负向词频表
    pos_words = []  # 正向词列表
    _pos_dict = {}  # 正向词频表

    global sum_words_neg, sum_words_pos, neg_sents_train, pos_sents_train, stopwords

    # 读入stopwords
    with open("./stopwords.txt", encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = mystrip(stopwords)

    # 收集训练集正、负向的句子
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            if line[0] == "0":
                neg_sents_train.append(line[1:])
            else:
                pos_sents_train.append(line[1:])

    # 收集测试集正、负向的句子
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            if line[0] == "0":  #
                neg_sents_test.append(line[1:])
            else:
                pos_sents_test.append(line[1:])

    # 获得负向训练语料的词列表neg_words
    for i in range(len(neg_sents_train)):
        words = jieba.lcut(neg_sents_train[i])
        words = remove_stopwords(words)
        neg_words.extend(words)

    # 获得负向训练语料的词频表_neg_dict
    for word in set(neg_words):
        _neg_dict[word] = neg_words.count(word)
    sum_words_neg = len(neg_words)

    # 获得正向训练语料的词列表pos_words
    for i in range(len(pos_sents_train)):
        words = jieba.lcut(pos_sents_train[i])
        words = remove_stopwords(words)
        pos_words.extend(words)

    # 获得正向训练语料的词频表_pos_dict
    for word in set(pos_words):
        _pos_dict[word] = pos_words.count(word)
    sum_words_pos = len(pos_words)

    # 返回训练得到的词频表
    return _neg_dict, _pos_dict


if __name__ == "__main__":
    # 统计训练集：
    neg_dict, pos_dict = my_init()  # 接收返回的词典

    rights = 0  # 记录模型正确分类的数目
    neg_dict_keys = neg_dict.keys()
    pos_dict_keys = pos_dict.keys()

    print(len(neg_sents_train))
    print(len(pos_sents_train))

    # 先验概率
    prior_neg = len(neg_sents_train) / (len(neg_sents_train) + len(pos_sents_train))
    prior_pos = len(pos_sents_train) / (len(neg_sents_train) + len(pos_sents_train))

    # print(prior_neg)
    # print(prior_pos)

    # 测试：
    for i in range(len(neg_sents_test)):  # 用negative的句子做测试
        st = jieba.lcut(neg_sents_test[i])  # 分词，返回词列表
        st = remove_stopwords(st)  # 去掉停用词

        p_neg = np.log(prior_neg)  # Ci=neg的时候，目标函数的值
        p_pos = np.log(prior_pos)  # Ci=pos的时候，目标函数的值

        # 计算p_neg和p_pos
        for word in st:
            p_neg += np.log((neg_dict.get(word, 0) + 1) / (sum_words_neg + len(neg_dict_keys)))
            p_pos += np.log((pos_dict.get(word, 0) + 1) / (sum_words_pos + len(pos_dict_keys)))

        if p_pos < p_neg:
            rights += 1

    for i in range(len(pos_sents_test)):  # 用positive的数据做测试
        st = jieba.lcut(pos_sents_test[i])
        st = remove_stopwords(st)

        p_neg = np.log(prior_neg)  # Ci=neg的时候，目标函数的值
        p_pos = np.log(prior_pos)  # Ci=pos的时候，目标函数的值

        # 计算p_neg和p_pos
        for word in st:
            p_neg += np.log((neg_dict.get(word, 0) + 1) / (sum_words_neg + len(neg_dict_keys)))
            p_pos += np.log((pos_dict.get(word, 0) + 1) / (sum_words_pos + len(pos_dict_keys)))

        if p_pos >= p_neg:
            rights += 1

    print("准确率:{:.1f}%".format(rights / (len(pos_sents_test) + len(neg_sents_test)) * 100))
