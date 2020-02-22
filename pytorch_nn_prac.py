# nn package
# 1. convnet
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTconvnet(nn.Module):
    def __init__(self):
        super(MNISTconvnet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5) # input = 1, output = 10, kernel size =5x5
        self.pool1 = nn.MaxPool2d(2,2) # kernel size =2, stride =2
        self.conv2 = nn.Conv2d(10,20,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(320,50) # in_features = 320, out_features = 50
        self.fc2 = nn.Linear(50,10) # out_feature = 10 개

    def forward (self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1) #
        # x =torch.randn(4,4)
        # print(x.size()) = torch.Size([4,4])
        # y = x.view(16) # 뷰는 같은 개수와 element의 다른 모양으로 바꾸는!
        # y.size() = torch.Size([16])
        # z = x.view(-1, 8) # -1은 다른 차원에서 부터 추론된다 아 맞네 개수는 같아야 하므로!
        # z.size() = torch.Size([2,8])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Conv2d : nSamples(배치사이즈?) x nChannels x Height x Width 4D tensor 이용
net = MNISTconvnet()
# print(net)

# create fake 이미지
input = torch.randn(1,1,28,28)
out = net(input)
print(out.size()) # torch.Size([1, 10])

target = torch.tensor([3], dtype = torch.long)
print(target) # tensor([3])
loss = nn.CrossEntropyLoss()
error = loss(out, target)
error.backward()

print(error)

## Recurrent net

class RNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self. hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size) # input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1) # (data와 last_hidden) 을 콘캣 ->  이게 recursive
                                                  # (2x3)을 cat 0 이면 (4,3) 1이면 (2,6) 으로 콘캣한다
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output

rnn = RNN(50, 20, 10)

loss_fn = nn.MSELoss()

batch_size = 10
timestep = 5

# create fake 이미지
batch = torch.randn(batch_size, 50)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 10)

loss = 0
for t in range(timestep):
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()