# 1) Tokenization : corpus를 전처리 할때, 토큰화-정제(cleaning)-정규화(Normalization)
# 그 중 tokenization은 corpus를 의미있는 단위인 token 토큰으로 나누는 작업
# 토큰의 기준은 다양한데, 기준을 word로 하는경우, 단어 토큰화(word tokenization)이라고 한다
# 여기서 단어는 진짜 단어 이외 단어구, 의미를 가지는 문자열도 된다.

# 구두점(punctuation)같은 문자,온점, 반점, 세미콜론, 느낌표는 제외하는 과정을 해보자.
# input : Time is illustion. Lunch time double so! 이를 구두점 제외
# output : "Time", "is", "illustion", "Lunchtime","double", "so"
#-> 구두점을 지운 후 whitespace(띄어쓰기) 기준으로 잘라냄
# 보통 토큰화는 단순히 구두점이나 특수문자를 제거하는 정제(cleaning)로 해결되지 않음
# 오히려 토큰이 의미를 잃을 수 있다

# 어퍼스트로피(')를 어떻게 토큰화 할 것인가? ex) don't Jone's
# don't / don t / dont / do n't && jone's / jone s / jone / jones 가있다

# word_tokenizer 와 WordPunctTokenizer를 이용해 어퍼스트로피 확인해보자!
#NLTK는 영어 corpus를 토큰화하는 도구 제공
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))


# 토큰화에서 고려해야할 사항 - 생각보다 섬세해야 함
# 1. 구두점이나 특수문자를 단순제외해서는 안된다
# 온점은 문장 경계를 알 수 있다. ph.D, $60(60달러),AT&T나 45.5, 123,456,789원 같은 표현
# 2. 줄임말과 단어 내에 띄어쓰기가 있는 경우
# 영어에서 어퍼스트로피는 압축된 단어를 펼치는 역할 : I'm = I am , What're = what are등
# New York은 한 단어이지만 사이에 띄어쓰기 존재 이런 단어들을 하나로 인식할 수 있는 능력 필요
# 3. 표준 토큰화 예제
# ex) Penn Treebank Tokenization의 규칙
# (1) 하이픈으로 구성된 단어는 하나이다.
# (2) doesn't같은 어퍼스트로피로 '접어'가 같이있는 단어는 분리해준다.

from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text1 = "starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own"
print (tokenizer.tokenize(text1))

# 문장 토큰화(sentence tokenization)
# 토큰의 단위가 문장일때 어떻게?
# corpus 내에서 문장단위로 구분하는 작업 = sentence segmentation이라고도 한다
# corpus가 cleaning되지 않은 상태라면 문장단위로 구분되어 있지 않을 가능성이 있다.
# 이 corpus를 이용하려면 문장 토큰화가 필요하다
# 온점이나 느낌표, 물음표로 구분할 수 있지만 그렇지 않을 수 있다 .은 어디든 나올 수 있다
# ex) Get into server IP 192.168.56.31 and save the log file. 같은 문장
# 그러므로 다른 규칙을 이용해야 한다

from nltk.tokenize import sent_tokenize # 문장 토큰화를 해주는 기능
text2="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
print(sent_tokenize(text2))

text3 = "Get into server IP 192.168.56.31 and save the log file."
print(sent_tokenize(text3))

# 한국어 일때는?
import kss
text4 = "딥러닝 자연어 처리가 재미있기는 합니다.그런데 문제는 영어보다 한국어로 할때 너무 어려워요. 농담아니에요."
print(kss.split_sentences(text4))

# Binary Classifier = 이진분류기
# 문장 토큰화에서 예외사항을 발생시키는 온점(.)을 처리하기 위해 입력에 따라 2개 클래스로 분류
# class1 = 온점이 단어의 일부분인 경우 / class2 = 온점이 문장의 구분자인 경우
# 머신러닝을 이용하거나 규칙을 넣어서 알고리즘으로 2진분류기를 구현하기도 한다

# 한국어가 토큰화의 어려운 이유
# 1. 우리는 띄어쓰기도 단어단위가 아니고 어절단위(조사있음)이다 ->어절은 토큰화 지양 so 형태소 토큰화를 해야함
# 2.띄어쓰기가 영어보다 잘 이루어지지 않는다

# 품사태깅 = part-of-speech tagging
# 품사에 따라 표기가 같은 단어가 다른 의미를 가진다. ex) 못한다 할때 '못'과 못과 망치할때 '못'
# 어떤 품사로 쓰였는지가 중요-> 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는디 구분하는것이 품사태깅이다.
# KoNLPy를 이용해 품사태깅 확인가능

text5 = "I'm actively looking for Ph.D. students. And ou are a Ph.D. student."
print(word_tokenize(text5))

from nltk.tag import pos_tag
x = word_tokenize(text5)
pos_tag(x)

# 형태소 = morpheme tokenize : 한국어 -> 사용가능한 library: Okt(Open Korea Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)
from konlpy.tag import Okt
okt = Oky()
text6 = "대학원 생활은 정말 너무 힘든 것이 아니야"
print(okt.morphs(text6)) # 형태소 추출
print(okt.pos(text6)) # 품사 태깅
print(okt.nouns(text6)) # 명사 추출

from konlpy.tag import Kkma
kkma = Kkma()
print(kkma.morphs(text6))
print(kkma.pos(text6))
print(kkma.nouns(text6))
# 위와 결과는 다르다..