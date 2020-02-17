import pickle as pkl
from hazm import *
import numpy as np
from math import floor

alphas = 'ابپتسجچهخدذرزژسشصضطظعغفقکگلمنوهییٔءآاًهٔة _:,.;"123456789-)(?/'


def permissible_chars(word):
    for char in word:
        if char in alphas:
            return True

    return False



def spelitCorpus(corpus, trainRate=1.0, validationRate=0.0, testRate=0.0):
    if int(trainRate + validationRate + testRate) == 1:
        np.random.shuffle(corpus)
        trainSplitIndex = floor(len(corpus) * trainRate)
        validationSplitIndex = floor(len(corpus) * validationRate)
        training = corpus[:trainSplitIndex]
        validating = corpus[trainSplitIndex:trainSplitIndex + validationSplitIndex]
        testing = corpus[trainSplitIndex + validationSplitIndex:]

        return training, validating, testing
    else:
        try:
            raise ValueError('Wrong ratio.')
            raise Exception('The sumation of rates is not equal to 1 :(')
        except Exception as error:
            print('Caught this error: ' + repr(error))


W2I = {}
I2W = {}

with open('./dataset2/w2i.pkl', 'rb') as wi:
    W2I = pkl.load(wi)

vocab = []
for w, i in W2I.items():
    if (w not in vocab):
        vocab.append(w)
print("vocab len = " + str(len(vocab)))
# with open('./dataset/i2w.pkl','rb') as iw:
#     I2W = pkl.load(iw)

def checinvocab(words):
    for word in words:
        if word not in vocab:
            print(word)
            return False
    return True
c =0
dataset = []
with open('./dataset/data.txt', 'r') as dfile:
    lines = dfile.readlines()
    for line in lines:
        iline = []
        words = word_tokenize(line)
        if (len(words) <= 50 and checinvocab(words)):
            for word in words:
                if permissible_chars(word):
                    iline.append(W2I[word])
            c +=1
            dataset.append(iline)
            print(c)


dataset = np.array(dataset)

train, validate, test = spelitCorpus(dataset)

with open("./dataset2/train.txt", "w") as trainfile:
    for senItem in train:
        sentence = ''
        for ind in senItem:
            sentence = sentence + ' ' + str(ind)
        trainfile.write(sentence + '\n')

with open("./dataset2/val.txt", "w") as valfile:
    for senItem in validate:
        sentence = ''
        for ind in senItem:
            sentence = sentence + ' ' + str(ind)
        valfile.write(sentence + '\n')

with open("./dataset2/test.txt", "w") as testfile:
    for senItem in test:
        sentence = ''
        for ind in senItem:
            sentence = sentence + ' ' + str(ind)
        testfile.write(sentence + '\n')
