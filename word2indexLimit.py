from hazm import *
import pickle as pkl
import numpy as np

alphas = 'ابپتسجچهخدذرزژسشصضطظعغفقکگلمنوهییٔءآاًهٔة _:,.;"123456789-)(?/'

class GetOutOfLoop( Exception ):
    pass


def permissible_chars(word):

    for char in word:
        if char in alphas:
            return True

    return False


dataSet = './dataset/data.txt'
Vocabulary = []
c = 0
try:
    with open(dataSet) as dts:
        lines = dts.readlines()
        for line in lines:
            words = word_tokenize(line)
            if len(words) <= 50:
                c += 1
                for word in words:
                    if (word not in Vocabulary and permissible_chars(word)):
                        print(word)
                        Vocabulary.append(word)
                        if(len(Vocabulary)==10000):
                            raise GetOutOfLoop
except GetOutOfLoop:
    pass


print("Vocab size = "+str(len(Vocabulary)))
print(c)
W2I = {}
I2W = {}
for idx, word in enumerate(Vocabulary):
    W2I[word] = idx
    I2W[idx] = word

w2iOutPut = open('./dataset2/w2i.pkl', 'wb')
pkl.dump(W2I, w2iOutPut)
w2iOutPut.close()


i2wOutPut = open('./dataset2/i2w.pkl', 'wb')
pkl.dump(I2W, i2wOutPut)
i2wOutPut.close()
