import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib as plt

dtype = torch.FloatTensor
sentence =['ich mochte ein bier P', 'S i want a beer','i want a beer E']
# S : Symbol that shows starting point of decoding input
# E : Symbol that shows starting of decoding output
# P : Symbol that will fill in blank sequence if current batch data size is short than time steps

# Transformer parameter
src_voca = {'P':0, 'ich':1, 'mochte':2, 'ein':3, 'bier':4} # 딕셔너리 , P는 padding = 0
src_voca_size = len(src_voca)

tgt_voca = {'P':0, 'i':1, 'want':2, 'a':3, 'beer':4, 'S':5, 'E':6}
number_dict = {i:w for i, w in enumerate(tgt_voca)}
tgt_voca_size = len(tgt_voca)

src_len = 5
tgt_len = 5

d_model = 512 # 임베딩 사이즈
d_ff = 2048 # feedforward dimension
n_layers = 6 # encoder decoder 층 개수
n_head = 8 # multi-head attention ㅐ헤드 개수
d_k = d_v = 64 # K = Q(같아야 한다) 디멘션 개수 ,V

def make_batch(sentence):
    input_batch = [[src_voca[n] for n in sentence[0].split()]] # list로 만듦
    output_batch = [[tgt_voca[n] for n in sentence[1].split()]]
    target_batch = [[tgt_voca[n] for n in sentence[2].split()]]
    return Variable(torch.LongTensor(input_batch)), Variable(torch.LongTensor(output_batch)), Variable(torch.LongTensor(target_batch))
    # Variable = autograd : 디폴트 requires_grad = False, tensor로 정의된 모든 API를 지원한다
    # x = Variable(torch.ones(2,2), requries_grad = True) 일때
    # 모델 파라미터 x를 학습하기위해 loss함수로 계산된 loss를 저장하기 위해 variable loss사용
    # ∂loss/∂x를 계산하는 loss.backward를 호출하면 pytorch는 x 변수에 gradient를 저장
    # requries_grad는 변수 x가 학습가능한지 를 나타냄! 즉, 위에꺼는 학습불가
def get_sinusoid_encoding_table(n_position, d_model): # positonal encoding
    def cal_angle(position, hid_idx):
        return position/np.power(10000, 2*(hid_idx // 2)/d_model) # 10000^(2i/d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)] # hid_j는 0-d_model까지

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])
    # x[startpoint:endpoint:skip] 시작점부터 skip의 차이씩 띄우면서 표현됨
    # ex) l = range(20)
    # l[1::3] = [1,4,7,10,13,16,19] 이런식으로 표현됨
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:,1::2])
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # eq : element-wise equality
    # x = torch.tensor([1,2,3,4])    # dim = 1
    # torch.unsqueeze(x,0) = tensor([[1,2,3,4]])
    # torch.unsqueeze(x,1) = tnesor([[1],
    #                                [2],
    #                                [3],
    #                                [4]])
    return pad_attn_mask.expand(batch_size, len_q, len_k)
    # x = torch.tensor([[1],[2],[3]])
    # x.size() = torch.size([3,1])
    # x.expand(3,4) = tensor([1,1,1,1],[2,2,2,2],[3,3,3,3])
    # x.expand(-1,4) = tensor([1,1,1,1],[2,2,2,2],[3,3,3,3]) # -1은 사이즈가 변하지 않는다는 뜻

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1) # k 번째 diagonal을 0으로 만든다 나머지는 1
    # np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1) 일때
    # array([[ 1,  2,  3],
    #        [ 4,  5,  6],
    #        [ 0,  8,  9],
    #        [ 0,  0, 12]]) 이런식으로 표현된다
    subsequent_mask = torch.from_numpy(subsequent_mask).byte() # numpy에서 torch로 텐서 버전을 바꾼다

    return subsequent_mask

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct,self).__init__()
        self.softmax = nn.Softmax(dim = -1) # softmax의 dim? 소프트맥스계산되는 디멘션
        # NLLLoss 에는 Logsoftmax를 사용한다
        self.const = np.sqrt(d_k) # d_k는?

    def forward(self, Q, K, V, att_mask): # att_mask는
        score = torch.matmul(Q,K.transpose(-1,-2))/self.const # tranpose: 주어진 dim0과 dim1이 서로 바꿘다
        score.masked_fill_(att_mask, -1e9) # masked!
        # masked_fill_(mask, value) mask는 boolean으로, 마스크가 true인 곳에 value를 채움
        attn = self.softmax(score) # attn = attention distribution
        context = torch.matmul(attn, V)
        return context, attn

############################################################
# self 란 무엇인가?
# class Foo:
#     def func1(): # 인자가 self가 아니어도 오류는 나지 않는다
#         print("fuckck")
#     def func2(self):
#         print("fuck!!")
# f = Foo() # 해당 클래스에 대한 인스턴스 생성
# f.func2()=> function 2가 정상적으로 프린트 된다 # 인스턴스 메소드 호출 -> func2의 메소드인자는 self뿐이므로 인풋 필요없다
#  메소드인 func2의 인자 self에 대한 값은 파이썬이 자동으로 넘겨주기 때문에 인풋필요없다
# f.func1() -> 에러가 난다 self 인자는 없지만 파이썬이 자동으로 값을 전달하기 때문에 발생

# class 내의 self는 클래스 자체를 나타내는 인스턴스이다!
############################################################

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__() # d_v = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_head) # n_head 번 병렬수행 # concat을 하기 때문에 d_k x n_head 이다
        self.W_K = nn.Linear(d_model, d_k * n_head) #
        self.W_V = nn.Linear(d_model, d_k * n_head)

    def forward(self,Q, K, V, att_mask): # 인코더는 QKV가 다똑같고, 디코더는 KV는 같구 Q는 다르다
        residual = Q
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1,2)

        att_mask = att_mask.unsqueeze(1).repeat(1, n_head, 1,1) # unsqueeze(1)은 col로 변환

        context, attn = ScaledDotProduct()(q_s, k_s, v_s, att_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_head * d_v)
        # contiguous[인접한]() : self 텐서와 같은 data를 가지고 있는 contiguous 텐서를 리턴
        # 텐서의 열이나 행을 삭제(?)
        output = nn.Linear(n_head*d_v, d_model)(context) # 콘캣된 애를 한번 더 가중치 행렬을 통과시킵니다
        return nn.LayerNorm(output + residual), attn

class PositionwiseFFNN(nn.Module):
    def __init__(self):
        super(PositionwiseFFNN, self).__init__()  # conv1d 는 무엇인가 2d와 뭐가 다른가...
        # W1 = d_model x d_ff
        self.linear1 = nn.Conv1d(in_channels = d_model, out_channels = d_ff, kernel_size=1)
        # W2 = d_ff x d_model
        self.linear2 = nn.Conv1d(in_channels = d_ff, out_channels = d_model, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, input):
        residual = input
        output = self.linear1(input.transpose(1,2))
        output = self.relu(output)
        output = self.linear2(output).transpose(1,2)
        return nn.LayerNorm(d_model)(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.PWfeedforward = PositionwiseFFNN()
    def forward(self, enc_input, enc_self_attn_mask):
        enc_output, attn = self.enc_self_attn(enc_input, enc_input, enc_input, enc_self_attn_mask)
        enc_output = self.PWfeedforward(enc_output)
        return enc_output, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.src_emb = nn.Embedding(src_voca_size, d_model)
        # Embedding : 임베딩을 하기위한 table이 있다
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze = True)
        self.layer = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self, enc_input):
        enc_output = self.src_emb(enc_input)+self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        enc_self_attn_mask = get_attn_pad_mask(enc_input, enc_input)
        enc_self_attns = []
        for layer in self.layer:
            enc_output, enc_self_attn = layer(enc_output, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn) # append = concat 같은 느낌
        return enc_output, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.PWfeedforward = PositionwiseFFNN()

    def forward(self, dec_input, enc_output, dec_self_attn_mask, dec_enc_attn_mask):
        dec_output, dec_self_attn = self.dec_self_attn(dec_input, dec_input, dec_input, dec_self_attn_mask)
        dec_output, dec_end_attn = self.dec_enc_attn(dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output = self.PWfeedforward(dec_output)
        return dec_output, dec_self_attn, dec_end_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_voca_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model), freeze = True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_input, enc_input, enc_output):
        dec_output = self.tgt_emb(dec_input)+pos_emb(torch.LongTensor([5,1,2,3,4]))
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_input, dec_input)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_input)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequent_mask),0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_input, enc_input)

        dec_self_attn_mask = get_attn_pad_mask(dec_input, enc_input)

        dec_self_attn, dec_enc_attn = [],[]
        for layer in self.layers:
            dec_output, dec_self_attn, dec_enc_attn = layer(dec_output, enc_output, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attn.append(dec_self_attn)
            dec_enc_attn.append(dec_enc_attn)
        return dec_output, dec_self_attn, dec_enc_attn, dec_enc_attn

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_voca_size, bias = False)
        self.softmax = nn.Softmax()
    def forward(self, enc_input, dec_input):
        enc_output, enc_self_attn = self.encoder(enc_input)
        dec_output, dec_self_attn, dec_enc_attn = self.decoder(dec_input, enc_input, enc_output)
        dec_logit = self.protjection(dec_output)
        return dec_logit.view(-1, dec_logit.size(-1)), enc_self_attn, dec_self_attn, dec_enc_attn

model = Transformer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(20):
    optimizer.zero_grad()
    enc_input, dec_input, target_batch = make_batch(sentence)
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_input, dec_input)
    loss = criterion(outputs, target_batch.contiguous().view(-1))
    print('Epoch:','%04d'%(epoch+1), 'cost = '.format(loss))
    loss.backward()
    optimizer.step()