# import numpy as np
# from hazm import word_tokenize
#
# alphas = 'ابپتسجچهخدذرزژسشصضطظعغفقکگلمنوهییٔءآاًهٔة'
#
#
# def permissible_chars(word):
#     for char in word:
#         if char in alphas:
#             return True
#
#     return False
#
# vocab = []
# with open('./dataset/data.txt','r') as dataset:
#     for line in dataset.readlines():
#         for word in word_tokenize(line):
#             if word not in vocab and permissible_chars(word):
#                 vocab.append(word)
#
#         break
#
# vocab = np.array(vocab)
# print(vocab.size)

import pickle as pkl
from hazm import *
import numpy as np
from math import floor
dfile = open('./dataset/data.txt', 'r')
lines = dfile.readlines()
lenline = {}
for line in lines:
    words = word_tokenize(line)
    for line in lines:
        words = word_tokenize(line)
        if(len(words) in lenline):
            lenline[str(len(words))] +=1
        else:
            lenline[str(len(words))] = 1


for key,value in lenline:
    print(key + " : "+ str)