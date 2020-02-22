# character- level RNN는 read series of character 한 글자씩 알파벳 하나씩
# trochtext에서 편한 함수를 사용하지 않고 preprocess 한다.

######################DATA PREPARING#################################
from __future__ import unicode_literals, print_function, division
from io import open # io : 다양한 타입의 input output을 다루는 파이썬모듈
import glob # pathname 경로이름을 특정 패턴과 매칭 파이썬 모듈
# 특정 디렉토리에 있는 파일이름을 모두 알아야 할때 사용
# glob.glob(pathname) : 디렉토리 안의 파일들을 읽고 리스트로 리턴

import os # os operating system 을 간단하게 사용할 수 있는 모듈
# 환경변수나 디렉토리, 파일등의 os자원을 제어할 수 있도록 해주는 모듈

def findFile(path): return glob.glob(path)

# print(findFile('data/names/*.txt')) # ['data/names\\Arabic.txt', 'data/names\\Chinese.txt',..이런식으로 18개 국가언어]

import unicodedata # unicode: 캐릭터 데이터베이스 제공 모든 캐릭터하나하나 특징을 unicode로 정의
import string

all_letters = string.ascii_letters + ".,;'" # 아스키 대문자+소문자
# print(all_letters) # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;'
n_letters = len(all_letters)

# unicode : 운영체제, 프로그램, 언어와 상관없이 문자마다 고유한 코드를 16비트로 제공
def unicodeToascii(s):
    return ''.join(  #str.join(iterable) : iterable하게 문자열이 concat 되어있을때 return string
        c for c in unicodedata.normalize('NFD',s) # ''안에 하나씩 캐릭터를 더함(?
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToascii('Ślusàrski')) # 왜 괄호안이 유니코드가 아니지..?

category_lines = {} # dictionary
all_categories = [] # list # name per language

# read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding= 'utf-8').read().strip().split('\n') # strip('')괄호안의 캐릭터를 제외하고 복사한다
                                                                         # split('')괄호안의 캐릭터를 기준으로 잘라서 list로 만들어줌 ->  파일에 있는 애들을 한줄씩 떼서 복사 lines 리스트에 저장
    return [unicodeToascii(line) for line in lines] # lines에 있는 line을 for문으로 아스키로 전환
                                                    # 리스트로 리턴
for filename in findFile('data/names/*.txt'): # 파일안에 있는 메모장 파일들을 (파일안에 한줄에 이름 1개씩 있다)
    category = os.path.splitext(os.path.basename(filename))[0] # filename의 맨 마지막 파일이름을 떼는 것이 basename(국가언어이름), 그 걸 텍스트로 뜯어 category라는 문자열로 만든다.
    all_categories.append(category) # list에 새로운 아이템 category를 끝에 더한다.
    lines = readLines(filename) # 아스키로 전환된 1줄들을 리스트로
    category_lines[category] = lines # 그걸 따로 '카테고리' 이름안의 리스트로 만듦

print(lines)
n_categories = len(all_categories)
print(all_categories) # all_카테고리는 나라 카테고리만 따로 있다
# ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']

# print(category_lines['Italian'][:5])

#######################Turning names into tnesors#####################

# 1개 letter를 표현하기 위해 1-hot vector 1xn_letters 를 사용- 벡터에서 1개만 1 나머지는 0
# 2d 매트릭스 = line_length x 1 x n_letters : 파이토치에서는 항상 배치 디멘션이 들어간다. 여기서 1는 배치사이즈가 1
import torch
import torch.nn as nn
import torch.nn.functional as F

# all_letters 에서 letter(1글자) 인덱스를 찾는다
def letterToindex(letter):
    return all_letters.find(letter) # find : 순서대로 되어닜는 문자열에서 lowest 인덱스를 리턴
# print(all_letters.find('A'))

def letterTotensor(letter):
    tensor = torch.zeros(1,n_letters) # n_letters = 모든 캐릭터(글자)의 개수
    tensor[0][letterToindex(letter)] = 1 # 그부분만 1로 = 원핫벡터
    return tensor

def lineTotensor(line):
    tensor = torch.zeros(len(line),1,n_letters) # line의 개수
    for li, letter in enumerate(line):
        # enumerate : 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate객체를 리턴
        # for i, name in enumerate(['dd','se','erds']):
        #   print(i, name) -> 0 dd, 1 se, 2 erds 이런식으로 순서값과 함께 출력된다
        tensor[li][0][letterToindex(letter)] = 1
    return tensor

# print(letterTotensor('J'))
# print(lineTotensor('Jones').size())
# print(lineTotensor('Jones'))

##############################Creating the network######################################
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden),1)
        hidden = self.i2h(combined)
        output = self.h2o(combined)
        output = self.softmax(output)
        return output, hidden # output size는 18,


    def initHidden(self): # 처음 콘캣하는 히든은 제로로 한다.
        return torch.zeros(1,self.hidden_size)
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
# input size = letter개수 , hidden size = n_hidden(=128), output size = 카테고리 수
# input = letterTotensor('A')
# hidden = torch.zeros(1,n_hidden)
#
# output, next_hidden = rnn(input, hidden)
# print(output) # tensor([[-2.9124, -2.8885, -2.9690, -2.8200, -2.8675, -2.8919, -2.9422, -2.9451,
#               # -2.9677, -2.8755, -2.8914, -2.8306, -2.8650, -2.8065, -2.8743, -2.8828,
#               # -2.8759, -2.9395]], grad_fn=<LogSoftmaxBackward>)

input = lineTotensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden) # slice를 사용
print(output) # tensor([[-2.9124, -2.8885, -2.9690, -2.8200, -2.8675, -2.8919, -2.9422, -2.9451,
              # -2.9677, -2.8755, -2.8914, -2.8306, -2.8650, -2.8065, -2.8743, -2.8828,
              # -2.8759, -2.9395]], grad_fn=<LogSoftmaxBackward>)
# 그러므로 두 결과는 같다
# 왜 한글자만 넣는거지?
# print(input[0].size())
# print(hidden.size())
# print(torch.cat((input,hidden),1)) # 왜 안되는것인가....
#########################Training#############################
# 1. preparing training

def categoryFromOutput(output):
    top_n, top_i = output.topk(1) # 주어진 텐서(output)에서 가장 큰 element를 (k=1)개만큼(?) 리턴한다
    category_i = top_i[0].item() # 1개 element를 가진 tensor의 value를 리턴
    return all_categories[category_i],category_i # 인덱스에 해당하는 값과 인덱스를 리턴

print(categoryFromOutput(output))

import random

def randomChoice(l): # l list에서 랜덤으로
    return l[random.randint(0,len(l)-1)] # 0에서 len-1까지 중에 한 integer의 list에 해당하는 값을 리턴

def randomTrainingExample():
    category = randomChoice(all_categories) # 랜덤으로 나라먼저 뽑음
    line = randomChoice(category_lines[category]) # 나라 안의 리스트에서 이름을 1개 랜덤 뽑음
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    # 카테고리를 텐서화 -> (target)정답
    line_tensor = lineTotensor(line) # 뽑은 이름을 텐서로 변환
    return category, line, category_tensor, line_tensor

# for i in range(10): # 10번 랜덤으로 뽑는다 -> 시험삼아 해본것(?)
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category = ', category, "/ line = ", line) # 뽑은 나라와 이름

# 2. train

criterion = nn.NLLLoss()

learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad() # 모델의 모든 파라미터를 0으로 세팅

    for i in range(line_tensor.size()[0]): # 샘플수만큼 반복
        output, hidden = rnn(line_tensor[i],hidden)

    loss = criterion(output, category_tensor) # category_tensor가 target tensor
    loss.backward() # back propagation

    for p in rnn.parameters(): # optimization하는 역할!
        p.data.add_(-learning_rate, p.grad.data) # add_ 는 inplace를 의미한다.
        # 즉, p.data = p.data + -lr*p.grad.data 이다!

        # = SGD와 같다 SGD식: theta = theta- rho*g (dJ/dtheta=그레디언트/ theta:모델의 매개변수 rho:lr g:그레이언트 )
        # 기계학습은 최적화(optimization)의 과정이다. -> 최적화 문제는 주로 미분을 사용한다.
        # 편미분(partial differentiation)= 변수하나하나에 대해서 미분)
        # 의 결과는 벡터인데 이 벡터를 gradient라 한다.
        # 편미분으로 다차원 공간에서 최저점을 찾을 수 있다
    return output, loss.item()

# 3. training

import time
import math

n_iter = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_loss = [] # plot하기 위해 모든 loss를 추적한다(리스트)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' %(m,s)

start = time.time()

for iter in range(1, n_iter+1): # 10만번 돌림
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter% print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category # 카테고리 = 정답
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter/n_iter *100, timeSince(start), loss, line, guess, correct))
                                            # 몇번돌았는지,  백분율,           시간,         로스, 이름,  추정, 맞았는지 여부

    if iter % plot_every ==0:
        all_loss.append(current_loss / plot_every) # all_lose에는 1000마다 1번씩 로스를 더 넣어준다
        current_loss = 0 # 이걸 왜 0으로 만들었지???

##############################plot##############################
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_loss)

########################Evaluating the result ################################
# 네트워크가 다른 카테고리에 대해 얼마나 잘 작동하는지를 알기위해 confusion matrix를 만들었다
# row는 진짜 언어 column은 추정한 언어

confusion = torch.zeros(n_categories, n_categories) # 추론이 맞는지 기록하기 위한
n_confusion = 10000

def evaluate(line_tensor): # train과 똑같지만 백프로파게이션은 안한다
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i]/confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

############user가 만든 인풋에서 돌려보기################
def predict(input_line, n_prediction = 3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineTotensor(input_line))

        topv, topi = output.topk(n_prediction, 1,True)
        prediction = []

        for i in range(n_prediction):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            prediction.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Kim')
predict('Satoshi')
predict('Pham')
predict('Ivanka')