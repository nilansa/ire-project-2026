"""Qrel-free, teacher-free selection from a dense-ranked 50-document pool."""
import hashlib
import random
import numpy as np

TASK = 'ire-disco-tttembed-selection-20261116-0700'

def seed(qid, label):
    return int(hashlib.sha256(f'{TASK}|{qid}|{label}'.encode()).hexdigest(), 16)

def select(qid, doc_ids, token_scores):
    """token_scores[i,j] is max cosine of query token j to tokens of document i."""
    scores = np.asarray(token_scores)
    if len(doc_ids) != 50 or len(set(doc_ids)) != 50 or scores.ndim != 2 or scores.shape[0] != 50 or scores.shape[1] == 0 or not np.isfinite(scores).all():
        raise ValueError('invalid pool, document IDs or contextual token scores')
    positive = np.maximum(scores, 0)
    best = np.zeros(scores.shape[1], dtype=positive.dtype)
    selected = []
    for _ in range(10):
        gains = np.maximum(positive - best, 0).sum(axis=1)
        gains[selected] = -np.inf
        index = int(np.argmax(gains))  # first index wins, i.e. original dense rank
        selected.append(index)
        best = np.maximum(best, positive[index])
    maxsim = sorted(range(50), key=lambda i: (-float(positive[i].sum()), i, str(doc_ids[i])))[:10]
    counts = [sum(lo <= i < lo + 10 for i in selected) for lo in range(0, 50, 10)]
    rankmatched = []
    for k in range(3):
        rng = random.Random(seed(qid, f'replicate={k}'))
        draw = []
        for b, n in enumerate(counts):
            draw.extend(rng.sample(range(10*b, 10*b+10), n))
        rankmatched.append(sorted(draw))
    uniform = sorted(random.Random(seed(qid, 'uniform=0')).sample(range(50), 10))
    groups = {'dense10': list(range(10)), 'coverage': selected, 'bert_maxsim': maxsim,
              'rankmatched_0': rankmatched[0], 'rankmatched_1': rankmatched[1],
              'rankmatched_2': rankmatched[2], 'uniform': uniform}
    assert all(len(a) == len(set(a)) == 10 and all(0 <= i < 50 for i in a) for a in groups.values())
    return {name: [doc_ids[i] for i in idx] for name, idx in groups.items()}

if __name__ == '__main__':
    ids = [str(i) for i in range(50)]
    x = np.zeros((50, 2), dtype=np.float32)
    x[0] = [.9, .9]
    x[1] = [.8, .8]
    x[10] = [.95, 0]
    x[20] = [0, .95]
    a = select('test', ids, x)
    assert a['coverage'][:3] == ['0', '10', '20']
    assert a['bert_maxsim'][:3] == ['0', '1', '10']
    for k in range(3):
        assert [sum(int(i)//10 == b for i in a[f'rankmatched_{k}']) for b in range(5)] == [sum(int(i)//10 == b for i in a['coverage']) for b in range(5)]
    assert select('test', ids, x) == a
    print('selector synthetic coverage, independent ranking, rank matching and deterministic seed: PASS')
