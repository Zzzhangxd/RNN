import torch
import glob
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"  # 所有ASCII字母以及一些常见的标点符号和空格在文本处理中频繁出现
n_letters = len(all_letters)


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    lines = open(filename).read().strip().split('\n')  # 读入所有行
    return [unicodeToAscii(line) for line in lines]


category_lines = {}  # 构建一个字典，存储每个类别对应的行内容列表
all_categories = []  # 存所有类别名称
for filename in findFiles('/home/zxd/桌面/nlp/data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]  # 得到每个文件的名字
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines  # category相当于标签，lines的内容相当于数据，类别名为键

n_categories = len(all_categories)


# 查找字母在all_letters中的索引，例如"a" = 0
# 相当于对词进行one_hot编码
def letterToIndex(letter):
    return all_letters.find(letter)


# one_hot编码
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        # 将对应位置的张量值设置为1
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
