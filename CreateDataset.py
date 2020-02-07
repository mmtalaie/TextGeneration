import pickle as pkl
from hazm import *
import numpy as np
from math import floor

alphas = 'ابپتسجچهخدذرزژسشصضطظعغفقکگلمنوهییٔءآاًهٔة'


def permissible_chars(word):
    for char in word:
        if char in alphas:
            return True

    return False


def spelitCorpus(corpus, trainRate=0.6, validationRate=0.2, testRate=0.2):
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

with open('./dataset/w2i.pkl', 'rb') as wi:
    W2I = pkl.load(wi)

# with open('./dataset/i2w.pkl','rb') as iw:
#     I2W = pkl.load(iw)
c =0
dataset = []
with open('./dataset/data.txt', 'r') as dfile:
    lines = dfile.readlines()
    for line in lines:
        iline = []
        words = word_tokenize(line)
        for word in words:
            if permissible_chars(word):
                iline.append(W2I[word])
        dataset.append(iline)



dataset = np.array(dataset)

train, validate, test = spelitCorpus(dataset)

with open("./dataset/train.txt", "w") as trainfile:
    for senItem in train:
        sentence = ''
        for ind in senItem:
            sentence = sentence + ' ' + str(ind)
        trainfile.write(sentence + '\n')

with open("./dataset/val.txt", "w") as valfile:
    for senItem in validate:
        sentence = ''
        for ind in senItem:
            sentence = sentence + ' ' + str(ind)
        valfile.write(sentence + '\n')

with open("./dataset/test.txt", "w") as testfile:
    for senItem in test:
        sentence = ''
        for ind in senItem:
            sentence = sentence + ' ' + str(ind)
        testfile.write(sentence + '\n')
