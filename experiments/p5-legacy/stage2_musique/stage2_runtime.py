import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DISCO_ROOT = ROOT / 'DISCo'
sys.path.insert(0, str(DISCO_ROOT))
sys.path.insert(0, str(ROOT))

from stage2_adapter import assert_qrel_blind, evidence_word_count, load_frozen_inputs, map_internal_ids


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def qid_order():
    return json.loads((ROOT / 'qid_order.json').read_text(encoding='utf-8'))


def doc_ids():
    return json.loads((ROOT / 'doc_id_map.json').read_text(encoding='utf-8'))


def corpus_lookup():
    corpus, _ = load_frozen_inputs()
    return {str(row['doc_id']): row for row in corpus}


def save_arm_outputs(arm, per_query_predictions, per_query_latency, evidence_lookup, latency_meta=None):
    qids = qid_order()
    if list(per_query_predictions) != qids:
        raise AssertionError(f'{arm}: query order mismatch')
    if latency_meta is None and list(per_query_latency) != qids:
        raise AssertionError(f'{arm}: latency qid order mismatch')
    for k in (2, 3, 5):
        pred = {qid: list(per_query_predictions[qid][:k]) for qid in qids}
        if any(len(docs) != k for docs in pred.values()):
            raise AssertionError(f'{arm} K={k}: expected exactly K predictions per question')
        words = {
            qid: sum(evidence_word_count(evidence_lookup[did]['title'], evidence_lookup[did]['text']) for did in pred[qid])
            for qid in qids
        }
        write_json_new(ROOT / 'outputs' / 'predictions' / f'{arm}_k{k}.json', pred)
        write_json_new(ROOT / 'outputs' / 'evidence_words' / f'{arm}_k{k}.json', words)
        if latency_meta is None:
            write_json_new(ROOT / 'outputs' / 'latency' / f'{arm}_k{k}.json', per_query_latency)
        else:
            write_json_new(ROOT / 'outputs' / 'latency' / f'{arm}_k{k}.json', latency_meta)


def load_runtime_symbols():
    import torch
    import src.retrievalmethods as rm
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
    rm.Indexer = Indexer
    rm.Searcher = Searcher
    rm.ColBERTConfig = ColBERTConfig
    rm.Run = Run
    rm.RunConfig = RunConfig
    torch.set_default_device('cpu')
    if torch.cuda.is_available() or torch.get_default_device().type != 'cpu':
        raise RuntimeError('CPU-only guard failed')
    return torch, rm


def make_config(method, k):
    from omegaconf import OmegaConf
    config = OmegaConf.load(DISCO_ROOT / 'configs' / 'retrieval.yaml')
    common = {
        'method': method,
        'k': k,
        'index': False,
        'overwrite_index': False,
        'data': {'dataset_name': 'musique', 'loader_type': 'beir', 'query_type': 'forum'},
        'embedder': {'mode': 'disk', 'mv_type': 'colbertv2-plaid', 'type': 'BERT', 'emb_dim': 128, 'num_queries': None},
        'colbert': {'nbits': 2},
    }
    if method == 'baseline':
        common.update({'augment': False, 'k': 5, 'colbert_topk': 5})
    elif method == 'disco':
        common.update({
            'augment': True,
            'dbl_norm': True,
            'num_rh_augment': 8,
            'colbert_topk': 1,
            'disco': {'rerank_internal': True, 'rerank_external': True},
        })
    else:
        raise ValueError(method)
    return OmegaConf.merge(config, OmegaConf.create(common))


def run_baseline():
    torch, rm = load_runtime_symbols()
    from colbert.infra import Run
    os.chdir(DISCO_ROOT)
    assert_qrel_blind(DISCO_ROOT / 'data' / 'musique' / 'qrels' / 'test.tsv')
    qids = qid_order()
    id_map = doc_ids()
    evidence = corpus_lookup()
    config = make_config('baseline', 5)
    obj = rm.ColBERTBaseline(config)
    obj._init_searcher()
    qrel_rows = assert_qrel_blind(DISCO_ROOT / 'data' / 'musique' / 'qrels' / 'test.tsv')
    embed_start = time.perf_counter()
    obj.embedder.embed_full_dataset(obj.dataloader, mode=config.embedder.mode)
    embed_seconds = time.perf_counter() - embed_start
    qembs, qmasks = obj.embedder.qembs, obj.embedder.qmasks
    if qembs.shape[0] != len(qids):
        raise AssertionError('baseline query embeddings do not match frozen qid order')
    ranking_path = ROOT / 'outputs' / 'checkpoints' / 'colbert_top5.jsonl'
    if ranking_path.exists():
        raise FileExistsError(ranking_path)
    rankings = {}
    latencies = {}
    with torch.inference_mode():
        for i, qid in enumerate(qids):
            start = time.perf_counter()
            ranking = obj.searcher.search_all_modified(
                rm.DummyQueryForColbert(1), Q=qembs[i:i+1], M=qmasks[i:i+1],
                k=5, normalize_q=config.colbert.normalize_q,
            ).todict()
            search_seconds = time.perf_counter() - start
            rows = ranking.get(0, ranking.get('0'))
            if rows is None or len(rows) < 5:
                raise RuntimeError(f'baseline returned fewer than 5 documents for {qid}')
            internal = [int(row[0]) for row in rows[:5]]
            ranked = map_internal_ids(internal, id_map)
            rankings[qid] = ranked
            latencies[qid] = embed_seconds / len(qids) + search_seconds
            append_jsonl(ranking_path, {'qid': qid, 'doc_ids': ranked, 'latency_s': latencies[qid]})
    save_arm_outputs('colbert', rankings, latencies, evidence)
    write_json_new(ROOT / 'outputs' / 'runs' / 'colbert.json', {
        'method': 'baseline', 'nbits': 2, 'dataset': 'musique', 'qrel_rows': qrel_rows,
        'query_embedding_seconds': embed_seconds, 'index_load_seconds_excluded': True,
        'latency_definition': 'shared query-encoding time divided evenly across qids plus per-query search wall time',
        'n_queries': len(qids), 'verifier_calls': 0,
    })
    del obj


def run_disco(k):
    torch, rm = load_runtime_symbols()
    os.chdir(DISCO_ROOT)
    qrels_path = DISCO_ROOT / 'data' / 'musique' / 'qrels' / 'test.tsv'
    qrel_rows = assert_qrel_blind(qrels_path)
    qids = qid_order()
    id_map = doc_ids()
    evidence = corpus_lookup()
    config = make_config('disco', k)
    obj = rm.DISCo(config)
    init_start = time.perf_counter()
    obj._init_searcher()
    obj._init_base_searcher()
    index_load_seconds = time.perf_counter() - init_start

    timing = {'embedding_s': 0.0, 'search_by_query': [], 'shared_rerank_s': []}
    original_embed = obj.embedder.embed_full_dataset
    original_search = obj.search
    original_helper = obj._search_helper
    original_get_corpus = obj.embedder.get_corpus
    original_partial = rm.partial_chamfer_sim_batched_with_rerank
    qrel_rows = assert_qrel_blind(qrels_path)
    checkpoint = ROOT / 'outputs' / 'checkpoints' / f'disco_k{k}.jsonl'
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.open('x').close()
    embed_start = [None]
    shared_start = [None]

    def timed_embed(*args, **kwargs):
        start = time.perf_counter()
        result = original_embed(*args, **kwargs)
        timing['embedding_s'] += time.perf_counter() - start
        return result

    def timed_search(qembs, search_k, opt_vec):
        per_query = [0.0] * len(qembs)
        cursor = [0]
        def timed_helper(query, query_opt, helper_k):
            idx = cursor[0]
            start = time.perf_counter()
            result = original_helper(query, query_opt, helper_k)
            elapsed = time.perf_counter() - start
            per_query[idx] = elapsed
            append_jsonl(checkpoint, {'phase': 'candidate_search', 'iteration': len(timing['search_by_query']) + 1, 'qid': qids[idx], 'elapsed_s': elapsed, 'completed': True})
            cursor[0] += 1
            return result
        obj._search_helper = timed_helper
        try:
            result = original_search(qembs, search_k, opt_vec)
        finally:
            obj._search_helper = original_helper
        if cursor[0] != len(qembs):
            raise AssertionError('DISCo search did not complete every query')
        timing['search_by_query'].append(per_query)
        return result

    def timed_get_corpus(*args, **kwargs):
        shared_start[0] = time.perf_counter()
        return original_get_corpus(*args, **kwargs)

    def timed_partial(*args, **kwargs):
        result = original_partial(*args, **kwargs)
        if shared_start[0] is not None:
            timing['shared_rerank_s'].append(time.perf_counter() - shared_start[0])
            shared_start[0] = None
        return result

    obj.embedder.embed_full_dataset = timed_embed
    obj.search = timed_search
    obj.embedder.get_corpus = timed_get_corpus
    rm.partial_chamfer_sim_batched_with_rerank = timed_partial
    obj._init_searcher = lambda: None
    obj._init_base_searcher = lambda: None
    try:
        run_start = time.perf_counter()
        result_ids, _ = obj.run()
        total_run_seconds = time.perf_counter() - run_start
    finally:
        rm.partial_chamfer_sim_batched_with_rerank = original_partial
    if result_ids.shape != (len(qids), k):
        raise AssertionError(f'DISCo K={k} returned unexpected shape {tuple(result_ids.shape)}')
    if len(timing['search_by_query']) != k or len(timing['shared_rerank_s']) != k:
        raise AssertionError('DISCo timing checkpoints do not cover every selection iteration')
    internal_predictions = result_ids.cpu().tolist()
    predictions = {qid: map_internal_ids(internal_predictions[i], id_map) for i, qid in enumerate(qids)}
    shared_per_query = sum(timing['shared_rerank_s']) / len(qids)
    latencies = {
        qid: timing['embedding_s'] / len(qids)
        + sum(timing['search_by_query'][step][i] for step in range(k))
        + shared_per_query
        for i, qid in enumerate(qids)
    }
    write_json_new(ROOT / 'outputs' / 'predictions' / f'disco_k{k}.json', predictions)
    words = {
        qid: sum(evidence_word_count(evidence[did]['title'], evidence[did]['text']) for did in predictions[qid])
        for qid in qids
    }
    write_json_new(ROOT / 'outputs' / 'evidence_words' / f'disco_k{k}.json', words)
    write_json_new(ROOT / 'outputs' / 'latency' / f'disco_k{k}.json', latencies)
    append_jsonl(checkpoint, {'phase': 'run_complete', 'k': k, 'n_queries': len(qids), 'total_run_seconds': total_run_seconds})
    write_json_new(ROOT / 'outputs' / 'runs' / f'disco_k{k}.json', {
        'method': 'disco', 'k': k, 'dataset': 'musique', 'qrel_rows': qrel_rows,
        'num_rh_augment': 8, 'colbert_topk': 1, 'nbits': 2, 'augment': True, 'dbl_norm': True,
        'rerank_internal': True, 'rerank_external': True, 'repo_defaults_otherwise': True,
        'index_load_seconds_excluded': index_load_seconds,
        'query_embedding_seconds': timing['embedding_s'], 'total_run_seconds': total_run_seconds,
        'latency_definition': 'shared query-encoding time evenly apportioned plus per-query candidate search and batched rerank wall time apportioned evenly',
        'n_queries': len(qids), 'verifier_calls': 0,
        'duplicate_docid_queries': [qid for qid in qids if len(set(predictions[qid])) < k],
    })
    del obj


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('baseline', 'disco2', 'disco3', 'disco5'):
        raise SystemExit('usage: stage2_runtime.py baseline|disco2|disco3|disco5')
    (ROOT / 'outputs' / 'predictions').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs' / 'evidence_words').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs' / 'latency').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs' / 'runs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs' / 'checkpoints').mkdir(parents=True, exist_ok=True)
    run = sys.argv[1]
    if run == 'baseline':
        run_baseline()
    else:
        run_disco(int(run[-1]))


if __name__ == '__main__':
    main()
