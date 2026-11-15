#!/usr/bin/env python3
"""Read-only alignment audit: no qrels or retrieval-method evaluation."""
import hashlib, json, os, sys, unicodedata
from collections import defaultdict
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import numpy as np

BASE = Path('/Users/nilan/.codex/scratch/cross-surface/ire-project')
PILOT = BASE / 'ire-ann-learning-pilot-20260907'
R8 = PILOT / 'experiment/official_r8'
DATA = PILOT / 'data/scifact'
TEACHER = BASE / 'ire-teacher-strength-20261031-001/shared/run-001'
SPLIT = BASE / 'ire-kd-positive-control-20261031-001/shared/run-002/inner_split.json'
COHORT = BASE / 'ire-feedback-value-sanity-20261031-001/shared/codex-execution-001/run-001/cohort.json'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def readlines(path):
    with path.open() as f: return [json.loads(x) for x in f if x.strip()]

def normalize(s): return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def audit():
    paths = [R8/'initial_embeddings.npz', R8/'frozen_candidate_ids.npz', R8/'frozen_candidate_scores.npz', R8/'frozen_candidate_query_ids.json', R8/'document_tie_keys.npy', R8/'source_final/run_pilot.py', R8/'source_final/ann_worker.py', DATA/'corpus.jsonl', DATA/'queries.jsonl', TEACHER/'teacher_scores_train505.npz', TEACHER/'provenance.json', SPLIT, COHORT]
    hashes = {str(p): sha(p) for p in paths}
    prov = json.loads((TEACHER/'provenance.json').read_text())
    for key, p in [('scifact_corpus', DATA/'corpus.jsonl'), ('scifact_queries', DATA/'queries.jsonl'), ('inner_split', SPLIT), ('cohort', COHORT), ('r8_candidate_ids', R8/'frozen_candidate_ids.npz'), ('r8_candidate_qids', R8/'frozen_candidate_query_ids.json')]:
        assert hashes[str(p)] == prov['inputs'][key]['sha256'], key
    assert hashes[str(TEACHER/'teacher_scores_train505.npz')] == prov['outputs']['teacher_scores_train505.npz']['sha256']
    corpus = readlines(DATA/'corpus.jsonl'); queries = readlines(DATA/'queries.jsonl')
    docs = [str(x['_id']) for x in corpus]; texts = {str(x['_id']): x['text'] for x in queries}
    assert len(docs)==len(set(docs))==5183 and len(texts)==len(queries)
    assert json.loads(COHORT.read_text())['document_id_row_order']==docs
    qids = list(map(str, json.loads((R8/'frozen_candidate_query_ids.json').read_text())))
    split = json.loads(SPLIT.read_text()); dev = list(map(str,split['train_prime_qids'])); ev = list(map(str,split['inner_qids']))
    parent = list(map(str,split['train_qids']))
    assert len(qids)==len(set(qids))==809 and len(dev)==405 and len(ev)==100 and set(dev).isdisjoint(ev) and set(dev)|set(ev)==set(parent) and len(set(parent))==505
    t = np.load(TEACHER/'teacher_scores_train505.npz', allow_pickle=False)
    tq = list(map(str,t['qid'])); td = t['doc_ids']
    assert tq==parent and len(set(tq))==505 and td.shape==(505,50)
    assert all(t[x].shape==(505,50) and np.isfinite(t[x]).all() for x in ('bge','medcpt'))
    e = np.load(R8/'initial_embeddings.npz',allow_pickle=False); c=e['corpus']; q=e['mine']
    assert c.shape==(5183,384) and q.shape==(809,384) and c.dtype==q.dtype==np.float32 and np.isfinite(c).all() and np.isfinite(q).all()
    ids=np.load(R8/'frozen_candidate_ids.npz',allow_pickle=False)['FaissFlatIP']; scores=np.load(R8/'frozen_candidate_scores.npz',allow_pickle=False)['FaissFlatIP']; keys=np.load(R8/'document_tie_keys.npy',allow_pickle=False)
    assert ids.shape==scores.shape==(809,50) and np.isfinite(scores).all() and len(keys)==5183 and len(set(map(int,keys)))==5183
    lookup={x:i for i,x in enumerate(qids)}
    assert all(x in lookup and x in texts for x in tq)
    for i,qid in enumerate(tq):
        assert [docs[int(j)] for j in ids[lookup[qid]]]==list(map(str,td[i])), f'teacher join failed {qid}'
    max_delta=0.; bad_set=[]; bad_order=[]; max_boundary=0.
    for i in range(809):
        row=ids[i]; assert len(set(map(int,row)))==50 and row.min()>=0 and row.max()<len(docs)
        vals=np.asarray(c @ q[i], dtype=np.float32)
        order=np.lexsort((keys,-vals)); expected=order[:50]
        delta=float(np.max(np.abs(vals[row]-scores[i]))); max_delta=max(max_delta,delta)
        if set(map(int,row))!=set(map(int,expected)):
            bad_set.append(qids[i]); max_boundary=max(max_boundary, float(vals[expected[-1]]-min(vals[row])))
        if not np.array_equal(row,expected): bad_order.append(qids[i])
    groups=defaultdict(list)
    for qid,text in texts.items(): groups[normalize(text)].append(qid)
    duplicates=[{'normalized_text':text,'query_ids':sorted(x)} for text,x in groups.items() if len(x)>1]
    moved=[]
    for g in duplicates:
        ds=set(g['query_ids'])&set(dev); es=set(g['query_ids'])&set(ev)
        if ds and es: moved.extend(x for x in ev if x in es)
    dev += moved; ev=[x for x in ev if x not in set(moved)]
    assert not {normalize(texts[x]) for x in dev}&{normalize(texts[x]) for x in ev}
    return {'source_hashes_sha256': hashes, 'teacher_rows_joined':len(tq), 'candidate_rows_checked':809,'max_candidate_score_abs_error':max_delta,'top50_set_mismatches':bad_set,'top50_order_mismatches':bad_order,'max_boundary_score_gap':max_boundary,'duplicate_groups':duplicates,'moved_inner_to_development':moved,'development_qids':dev,'inner_qids':ev,'split_counts':{'development':len(dev),'inner':len(ev)},'score_tolerance':2e-6,'status': 'PASS' if not bad_set and not bad_order and max_delta<=2e-6 else 'FAIL'}

if __name__=='__main__':
    result=audit(); print(json.dumps(result,indent=2,sort_keys=True)); sys.exit(0 if result['status']=='PASS' else 1)
