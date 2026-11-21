"""SCIDOCS prediction and separate qrel scoring; no downloads or implicit execution."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def parse_qrels(lines):
    reader = csv.DictReader(lines, delimiter='\t')
    if not reader.fieldnames or not {'query-id', 'corpus-id', 'score'} <= set(reader.fieldnames):
        raise ValueError('invalid qrel header')
    result = {}
    for row in reader:
        q, d = row['query-id'], row['corpus-id']
        if not q or not d or d in result.get(q, {}): raise ValueError('empty or duplicate qrel')
        result.setdefault(q, {})[d] = int(row['score'])
    return result


def read_qrels(archive):
    with zipfile.ZipFile(archive) as z:
        with z.open('scidocs/qrels/test.tsv') as f:
            return parse_qrels(io.TextIOWrapper(f, encoding='utf-8'))


def validate_slate(expected, actual, scores):
    if len(expected) != len(scores) or expected != actual or len(set(actual)) != len(actual):
        raise ValueError('feedback candidate mismatch')
    if not all(math.isfinite(x) for x in scores): raise ValueError('nonfinite teacher score')


def score(predictions, qrels, k):
    if set(predictions) != set(qrels): raise ValueError('prediction/qrel query IDs differ')
    per = {}
    for qid, rels in qrels.items():
        ranking = predictions[qid]
        if len(ranking) != len(set(ranking)): raise ValueError('duplicate document')
        positives = {did for did, rel in rels.items() if rel > 0}
        if not positives: raise ValueError('query has no positive judgments')
        per[qid] = len(set(ranking[:k]) & positives) / len(positives)
    return {'mean_recall': sum(per.values())/len(per), 'per_query': per}


def _jsonlines(z, name):
    with z.open(name) as f:
        for line in f: yield json.loads(line)


def shard_ids(ids, shard, count):
    if count <= 0 or not 0 <= shard < count or len(set(ids)) != len(ids):
        raise ValueError('invalid shard or duplicate query IDs')
    return ids[shard::count]


def batched_teacher(query, docs, predict_batch, batch_size=8):
    if not 1 <= batch_size <= 8:
        raise ValueError('teacher batch must be between 1 and 8')
    result = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        scores = predict_batch(query, batch)
        if len(scores) != len(batch) or not all(math.isfinite(x) for x in scores):
            raise ValueError('teacher output alignment failure')
        result.extend(scores)
    return result


def _model_files(path):
    path = Path(path)
    files = [p for p in path.iterdir() if p.is_file() and not p.name.startswith('.')]
    if not any(p.name in ('model.safetensors', 'pytorch_model.bin') for p in files):
        raise ValueError('model weights not present: '+str(path))
    return {p.name: digest(p) for p in sorted(files)}


def _write_new(path, data):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def predict(args):
    # Deliberately never opens qrels. Do not run before review and reservation.
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    start = time.perf_counter()
    archive_hash = digest(args.archive)
    if archive_hash != '96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1':
        raise ValueError('unexpected SCIDOCS archive hash')
    manifest = {'archive_sha256': archive_hash, 'retriever_files': _model_files(args.retriever),
                'teacher_files': _model_files(args.teacher), 'code_sha256': digest(__file__),
                'torch': torch.__version__, 'device': args.device}
    with zipfile.ZipFile(args.archive) as z:
        corpus = list(_jsonlines(z, 'scidocs/corpus.jsonl'))
        queries = list(_jsonlines(z, 'scidocs/queries.jsonl'))
    docs = {x['_id']: (x.get('title','')+' '+x.get('text','')).strip() for x in corpus}
    if len(docs) != len(corpus) or len(queries) != len({x['_id'] for x in queries}):
        raise ValueError('duplicate IDs')
    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.retriever, local_files_only=True)
    enc = AutoModel.from_pretrained(args.retriever, local_files_only=True).to(dev).eval()
    ttok = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher, local_files_only=True).to(dev).eval()
    def embed(texts):
        pieces = []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                t = tok(texts[i:i+args.batch_size], padding=True, truncation=True, max_length=512, return_tensors='pt').to(dev)
                hidden = enc(**t).last_hidden_state
                mask = t['attention_mask'].unsqueeze(-1)
                pieces.append(((hidden*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu())
        return torch.cat(pieces)
    docids = sorted(docs)
    tick = time.perf_counter(); vectors = embed([docs[d] for d in docids]).to(dev)
    queries_vec = embed([x['text'] for x in queries]).to(dev)
    encode_s = time.perf_counter() - tick
    if vectors.shape[0] < 126: raise ValueError('corpus too small')
    results = {}; feedback = {}
    for query, original in zip(queries, queries_vec):
        tick = time.perf_counter(); qid = query['_id']; similarities = vectors @ original
        if qid in docs: similarities[docids.index(qid)] = -torch.inf
        order = torch.argsort(similarities, descending=True, stable=True)[:125]
        ids = [docids[j] for j in order.tolist()]
        def teacher_batch(q, passages):
            t = ttok([q]*len(passages), passages, padding=True, truncation=True, max_length=512, return_tensors='pt').to(dev)
            with torch.no_grad(): return teacher(**t).logits.reshape(-1).cpu().tolist()
        raw_teacher = batched_teacher(query['text'], [docs[d] for d in ids], teacher_batch)
        validate_slate(ids, ids, raw_teacher)
        if len(raw_teacher) != 125: raise ValueError('teacher must emit one scalar per pair')
        feedback[qid] = {'docids': ids, 'scores': raw_teacher}
        v = vectors[order[:100]].detach(); raw = torch.tensor(raw_teacher[:100], device=dev)
        if raw.max() == raw.min(): raise ValueError('constant teacher scores')
        target = torch.softmax(((raw-raw.min())/(raw.max()-raw.min()))/2, dim=0).detach()
        current = original.detach().clone().requires_grad_(True)
        for _ in range(100):
            s = v @ current
            if (s.max()-s.min()).item() == 0: raise ValueError('constant student scores')
            normalized = (s-s.min())/(s.max()-s.min())
            loss = torch.nn.functional.kl_div(torch.log_softmax(normalized, dim=0), target, reduction='sum')
            grad, = torch.autograd.grad(loss, current)
            current = (current-0.005*grad).detach().requires_grad_(True)
        with torch.no_grad():
            updated = vectors @ current
            if qid in docs: updated[docids.index(qid)] = -torch.inf
            reretrieved = [docids[j] for j in torch.argsort(updated, descending=True, stable=True)[:100].tolist()]
        reranked = sorted(range(125), key=lambda i: (-raw_teacher[i], i))[:100]
        results[qid] = {'base': ids[:100], 'rerank125': [ids[i] for i in reranked], 'refit': reretrieved,
                        'logical_feedback_pairs': {'base':0,'rerank125':125,'refit':100},
                        'elapsed_seconds': time.perf_counter()-tick}
    _write_new(args.out, {'manifest':manifest, 'encode_seconds':encode_s, 'wall_seconds':time.perf_counter()-start,
                          'query_count':len(results), 'results':results, 'teacher_feedback':feedback,
                          'physical_teacher_pairs':125*len(results)})


def evaluate(args):
    with open(args.predictions, encoding='utf-8') as f: data = json.load(f)
    qrels = read_qrels(args.archive)
    out = {'prediction_sha256':digest(args.predictions), 'archive_sha256':digest(args.archive), 'methods':{}}
    for method in ('base','rerank125','refit'):
        out['methods'][method] = score({q:r[method] for q,r in data['results'].items()}, qrels, 100)
    _write_new(args.out, out)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict'); p.add_argument('--archive',required=True); p.add_argument('--retriever',required=True); p.add_argument('--teacher',required=True); p.add_argument('--out',required=True); p.add_argument('--device',default='cpu'); p.add_argument('--batch-size',type=int,default=16)
    e = sub.add_parser('score'); e.add_argument('--archive',required=True); e.add_argument('--predictions',required=True); e.add_argument('--out',required=True)
    args = parser.parse_args(); (predict if args.cmd == 'predict' else evaluate)(args)

if __name__ == '__main__': main()
