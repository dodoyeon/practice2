# 1. preparing text data
from __future__ import unicode_literals, print_function, division
from io import open
import os
import glob
import string
import torch
import torch.nn as nn
# import torch.nn.functional as F
import unicodedata
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def findFile(path): return glob.glob(path)

def uni2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

all_letters = string.ascii_letter +",.';"
n_letter = len(all_letters)

category_lines = {}
all_categories = []

def readLine(filename):
    lines = open(filename, encoding = 'utf-8').read().strip().split('\n')
    return [uni2ascii(line) for line in lines]

for filename in findFile('data/names/'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLine(filename)
    category_lines[category] = lines

n_category = len(all_categories)
# 2. name preprocassing
def letter2index(letter):
    return all_letters.find(letter)

def letter2tensor(letter):
    tensor = torch.zeros(1,n_letter)
    tensor[0][letter2index(letter)] = 1
    return tensor

def line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_letter)
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor

# 3. network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax(dim = 1) # 소프트맥스와 로그소프트는 둘다 인풋과 같은 디멘션을 낸다.

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden),1)
        hidden = self.i2h(combined)
        output = self.h2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)

n_hidden = 128
rnn = RNN(n_letter, n_hidden, n_category)
# test 해보기
input = letter2tensor('S')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input, hidden)
print(output)

# 4. training
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0,len(l)-1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tansor([all_categories.index(category)], dtype = torch.long)
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor

criterion = nn.NLLLoss()
learning_rate = 0.005

def train(target, input):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(input.size()):
        output, hidden = rnn(input[i], hidden)

    loss = criterion(output, target)
    loss.backward()

    for j in rnn.parameters(): # nn.Module.parameters : 모듈 파라미터를 리턴, 보통 optimizer에 pass한다
        j.data.add_(-learning_rate, j.grad.data) # grad: backward()를 한후에 tensor에 대한 grad가 계산된다

    return output, loss.item() # item() :  element 가 하나만 있는 tnesor에만 작용한다

n_iter = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_loss = []

for iter in range(1, n_iter+1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter%print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct ='o' if guess == category else 'x (%s)' %category
        print('%d %.4f / %s %s %s' % (iter, loss, line, guess, correct))

    if iter % plot_every == 0:
        all_loss.append(current_loss)
        current_loss = 0

criterion = nn.NLLLoss()
lr = 0.005
#
# def train (target, input):
#     hidden = rnn.initHidden()
#     rnn.zero_grad()
#     for i in range(input.size()):
#         output, hidden = rnn(input[i],hidden)
#     loss = criterion(output, target)
#     loss.backward()
#     for j in rnn.parameters():
#         j.data.add_(-lr,j.grad.data)
#     return output, loss.item()
#
# num_iter = 10000
# pr = 5000
# for iter range(1, num_iter+1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output, loss = train(category_tensor, line_tensor)
#     current_loss += loss
#
#     if iter%pr == 0:
#         # guess, guess_i = categoryFromOutput(output)
#         print(loss)

# 5.plot
plt.figure()
plt.plot(all_loss)

# 6. evaluating
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(1, n_iter)


# 7. predict

# class RNN_1(nn.Module):
#     def __init__(self, inputsize, hiddensize, outputsize):
#         super(RNN, self).__init__()
#         self.hiddensize = hiddensize
#
#         self.i2h = nn.Linear(inputsize+hiddensize, hiddensize)
#         self.h2o = nn.Linear(inputsize+hiddensize, outputsize)
#         self.softmax = nn.LogSoftmax(dim=1)
# # 그럼.. 아웃풋 결과가 디멘션1이라는건가?1개결과가나오도록?logsoftmax와 그냥 소프트맥스 차이는뭐지??
#
#
#     def forwad (self, input, hidden):
#         combined = torch.cat((input, hidden),1)
#         # 콘캣은 사이즈가 같지않아도 괜찮은가??input이랑 히든이 사이즈가 다른거같은디...
#         hidden = self.i2h(combined)
#         output = self.h2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#   def initHidden(self):
#       return torch.zeros(1, self.hiddensize)

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.h2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.Softmax(dim = 1)
#
#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden),1)
#         hidden = self.i2h(combined)
#         output = self.h2o(combined)
#         output = self.softmax(output)
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1,self.hidden_size)

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, 5) # 이 숫자들은 어떻게 정하는 것인가....
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(10,20,5)
#         self.pool2 = nn.MaxPool2d(2,2)
#         self.relu2 = nn.ReLU()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forwrad(self, input):
#         x = self.conv1(input)
#         x = self.pool1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.relu2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         output = self.relu2(x)
#
#         return output
#
# net = CNN()
# input = random.randint(1,1,28,28)
# output = net(input)
# target = torch.tensor([3], dtype = torch.long)
#
# criterion = nn.MSELoss()
# error = criterion(output, target)
# error.backward()
# print(error)