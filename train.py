import torch
from data import *
from model import *
import random
import time
import math


n_hidden = 128
n_epochs = 1000000
print_every = 5000
plot_every = 1000
learning_rate = 0.001


def categoryFromOutput(output):
    # 获取输出张量中值最大的那个元素及其索引
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    # 返回类别名称和索引
    return all_categories[category_i], category_i


# 从列表中随机选择一个元素
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingPair():
    category = randomChoice(all_categories)  # 随机选择一个类别
    line = randomChoice(category_lines[category])  # 从该类别中随机选择一行
    # 将类别转换为张量
    # 找到category在all_category里是第一个位置，索引是从0开始，第i个位置对应为第i个类别
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    # 将行转换为张量
    line_tensor = Variable(lineToTensor(line))
    # 返回类别名称，行，类别张量，行张量
    return category, line, category_tensor, line_tensor


# 初始化RNN模型，传入字母数量，隐藏层神经元数量和类别数量
rnn = RNN(n_letters, n_hidden, n_categories)
# 使用随机梯度下降
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
# 使用负对数似然损失函数
criterion = nn.NLLLoss()


# 训练函数，输入类别张量和行张量
def train(category_tensor, line_tensor):
    # 初始化隐藏层状态
    hidden = rnn.initHidden()
    # 梯度清零
    optimizer.zero_grad()

    # 逐个字母处理行张量
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    # 返回输出和损失值
    return output, loss.item()


# 初始化当前损失值
current_loss = 0
# 存储所有损失值的列表，用于绘图
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()


for epoch in range(1, n_epochs + 1):
    # 随机选择一个训练样本对
    category, line, category_tensor, line_tensor = randomTrainingPair()
    # 训练
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'yes' if guess == category else 'no (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')
