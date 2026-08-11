# Original code from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT
#
# Change in this file:
# - Added comments for clarity
# - Minor formatting (no functional changes)

import re
import logging
import numpy as np
import gensim.downloader as api

from tqdm import tqdm
from gensim.summarization import bm25

# Remove special tokens and split block into words
def remove_special_split(blk):
    return re.sub(r'</s>|<pad>|<s>|\W', ' ', str(blk)).lower().split()

# Initialize relevance scores using GloVe embeddings
def _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transforms=[], threshold=0.15):
    for transform_func in conditional_transforms:
        qbuf, dbuf = transform_func(qbuf, dbuf)

    dvecs = []
    for blk in dbuf:
        doc = [word_vectors[w] for w in remove_special_split(blk) if w in word_vectors]
        if len(doc) > 0:
            dvecs.append(np.stack(doc))
        else:
            dvecs.append(np.zeros((1, 100)))

    qvec = np.stack([word_vectors[w] for w in remove_special_split(qbuf) if w in word_vectors])

    scores = [np.matmul(qvec, dvec.T).mean() for dvec in dvecs]
    max_score_abs = max(scores) - min(scores) + 1e-6
    for i, blk in enumerate(dbuf):
        if 1 - scores[i] / max_score_abs < threshold:
            blk.relevance = max(blk.relevance, 1)

    return True

# Initialize relevance scores using BM25
def _init_relevance_bm25(qbuf, dbuf, conditional_transforms=[], threshold=0.15):
    for transform_func in conditional_transforms:
        qbuf, dbuf = transform_func(qbuf, dbuf)

    docs = [remove_special_split(blk) for blk in dbuf]
    model = bm25.BM25(docs)
    scores = model.get_scores(remove_special_split(qbuf))
    max_score = max(scores)
   
    if max_score > 0:
        for i, blk in enumerate(dbuf):
            if 1 - scores[i] / max_score < threshold:
                blk.relevance = max(blk.relevance, 1)
        return True

    return False

# Initialize relevance scores in the dataset
def init_relevance(dataset, method='glove', conditional_transforms=[]):
    print('Initialize relevance...')

    total = 0
    if method == 'glove':
        word_vectors = api.load("glove-wiki-gigaword-100")
        for qbuf, dbuf in tqdm(dataset):
            total += _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transforms)
    elif method == 'bm25':
        for qbuf, dbuf in tqdm(dataset):
            total += _init_relevance_bm25(qbuf, dbuf, conditional_transforms)
    else:
        pass

    print(f'Initialized {total} question-document pairs!')
