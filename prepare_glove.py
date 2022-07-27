'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

import numpy as np
import time
import json


def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


def load_w2v(embedding_dim, embedding_path, cpt_vocab):
    print('\nload embedding...')
    print(embedding_path)
    words = []
    for item in cpt_vocab:
        words.extend(item.split(' '))

    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))
    word_idx['unk'] = 0
    word_idx_rev[0] = 'unk'
    w2v = {}
    inputFile = open(embedding_path+'glove.6B.100d.txt', 'r', encoding='latin')
    inputFile.readline()
    for line in inputFile.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    # embedding = np.array(embedding)

    print("embedding.shape: {}:".format(len(embedding)))
    print("load embedding done!\n")

    # with open(embedding_path+'glove.json', 'w') as f:
    #     json.dump(embedding, f)

    return word_idx_rev, word_idx


def tokenize_glove(word_idx, str_input):
    tokens = str_input.split(' ')

    token_ids = []
    for token in tokens:
        if token in word_idx:
            token_ids.append(word_idx[token])
        else:
            token_ids.append(word_idx['unk'])
    return token_ids