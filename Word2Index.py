from hazm import *
import pickle as pkl


alphas = 'ابپتسجچهخدذرزژسشصضطظعغفقکگلمنوهییٔءآاًهٔة'


def permissible_chars(word):

    for char in word:
        if char in alphas:
            return True

    return False


dataSet = './dataset/data.txt'
Vocabulary = []
with open(dataSet) as dts:
    lines = dts.readlines()
    for line in lines:
        for word in word_tokenize(line):
            if (word not in Vocabulary and permissible_chars(word)):
                # print(word)
                Vocabulary.append(word)


print("Vocab size = "+str(Vocabulary.count))

W2I = {}
I2W = {}
for idx, word in enumerate(Vocabulary):
    W2I[word] = idx
    I2W[idx] = word

w2iOutPut = open('./dataset/w2i.pkl', 'w')
pkl.dump(W2I, w2iOutPut)
w2iOutPut.close()


i2wOutPut = open('./dataset/i2w.pkl', 'w')
pkl.dump(I2W, i2wOutPut)
i2wOutPut.close()
