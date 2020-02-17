import numpy as np
from hazm import word_tokenize

alphas = 'ابپتسجچهخدذرزژسشصضطظعغفقکگلمنوهییٔءآاًهٔة'


def permissible_chars(word):
    for char in word:
        if char in alphas:
            return True

    return False


vocab = []
max_size = 0
c = 0
with open('./dataset/data.txt', 'r') as dataset:
    for line in dataset.readlines():
        c = 0
        for word in word_tokenize(line):
            c += 1
            if word not in vocab and permissible_chars(word):
                vocab.append(word)
        print(c)
        print(line)
        if c > max_size: max_size = c



vocab = np.array(vocab)
print(vocab.size)
print(max_size)
