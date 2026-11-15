import hashlib
import json
import stat
from pathlib import Path

import numpy as np

from stage2_adapter import STAGE1, evidence_word_count, load_frozen_inputs
from stage2_metrics import bootstrap_indices, mean_ci, paired_difference_ci

ROOT = Path(__file__).resolve().parent
EXPECTED_GOLD_SHA256 = '084e5502ab3d61a368968c61c99e2949969e6a9bd40583fd9d35d99ad650185e'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_predictions_manifest():
    manifest = ROOT / 'PREDICTIONS.sha256'
    if not manifest.is_file():
        raise FileNotFoundError('predictions must be hashed before scoring')
    entries = []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.lstrip('* ')
        path = ROOT / rel
        if not path.is_file() or sha256(path) != digest:
            raise AssertionError(f'prediction hash mismatch: {rel}')
        entries.append(rel)
    expected = {f'outputs/predictions/{arm}_k{k}.json' for arm in ('bm25', 'colbert', 'disco') for k in (2, 3, 5)}
    if set(entries) != expected:
        raise AssertionError('prediction manifest does not cover exactly all nine arm/K files')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def score():
    verify_predictions_manifest()
    gold_path = STAGE1 / 'gold_HIDDEN.json'
    if stat.S_IMODE(gold_path.stat().st_mode) != 0o600:
        raise PermissionError('gold must be restored to mode 600 before scoring')
    gold_hash = sha256(gold_path)
    if gold_hash != EXPECTED_GOLD_SHA256:
        raise AssertionError('gold content hash changed')

    gold = load_json(gold_path)
    corpus, questions = load_frozen_inputs()
    qids = [str(q['id']) for q in questions]
    docs = {str(d['doc_id']): d for d in corpus}
    if set(gold) != set(qids):
        raise AssertionError('gold qids do not match the frozen sample')
    if any(len(set(gold[qid])) != 2 for qid in qids):
        raise AssertionError('expected exactly two distinct support doc_ids per question')

    prediction_data = {}
    evidence_data = {}
    latency_data = {}
    for arm in ('bm25', 'colbert', 'disco'):
        for k in (2, 3, 5):
            key = (arm, k)
            preds = load_json(ROOT / 'outputs' / 'predictions' / f'{arm}_k{k}.json')
            if list(preds) != qids or any(len(preds[qid]) != k for qid in qids):
                raise AssertionError(f'invalid predictions for {arm} K={k}')
            if any(did not in docs for values in preds.values() for did in values):
                raise AssertionError(f'unknown document id in {arm} K={k}')
            words = {qid: sum(evidence_word_count(docs[did]['title'], docs[did]['text']) for did in preds[qid]) for qid in qids}
            saved_words = load_json(ROOT / 'outputs' / 'evidence_words' / f'{arm}_k{k}.json')
            if saved_words != words:
                raise AssertionError(f'evidence word counts differ for {arm} K={k}')
            prediction_data[key] = preds
            evidence_data[key] = words
            latency_data[key] = load_json(ROOT / 'outputs' / 'latency' / f'{arm}_k{k}.json')

    draws = bootstrap_indices(len(qids))
    vectors = {}
    metrics = {'n_questions': len(qids), 'bootstrap_draws': 1000, 'bootstrap_seed': 0, 'arms': {}}
    for arm in ('bm25', 'colbert', 'disco'):
        metrics['arms'][arm] = {}
        for k in (2, 3, 5):
            preds = prediction_data[(arm, k)]
            complete = np.array([float(set(gold[qid]) <= set(preds[qid])) for qid in qids])
            recall = np.array([len(set(gold[qid]) & set(preds[qid])) / 2 for qid in qids])
            evidence = np.array([evidence_data[(arm, k)][qid] for qid in qids], dtype=float)
            lat_obj = latency_data[(arm, k)]
            latency = None
            if arm == 'bm25':
                if lat_obj.get('per_query_available'):
                    latency = np.array([lat_obj[qid] for qid in qids], dtype=float)
                else:
                    latency = None
            else:
                latency = np.array([lat_obj[qid] for qid in qids], dtype=float)
            vectors[(arm, k)] = {'complete_support': complete, 'support_recall': recall, 'evidence_words': evidence, 'latency_s': latency}
            metrics['arms'][arm][f'K{k}'] = {
                'complete_support': mean_ci(complete, draws),
                'support_recall': mean_ci(recall, draws),
                'evidence_words': mean_ci(evidence, draws),
                'latency_s': mean_ci(latency, draws) if latency is not None else {
                    'mean': lat_obj['mean_s'], 'p95': lat_obj['p95_s'], 'ci95': None,
                    'source': lat_obj['source'], 'per_query_available': False,
                },
            }

    comparisons = {'pairs': {}}
    pairs = (('disco', 'colbert'), ('disco', 'bm25'), ('colbert', 'bm25'))
    metric_names = ('complete_support', 'support_recall', 'evidence_words', 'latency_s')
    for k in (2, 3, 5):
        comparisons['pairs'][f'K{k}'] = {}
        for arm_a, arm_b in pairs:
            pair_key = f'{arm_a}-{arm_b}'
            comparisons['pairs'][f'K{k}'][pair_key] = {}
            for metric in metric_names:
                a = vectors[(arm_a, k)][metric]
                b = vectors[(arm_b, k)][metric]
                if a is None or b is None:
                    comparisons['pairs'][f'K{k}'][pair_key][metric] = {'available': False, 'reason': 'frozen BM25 per-query latency was not saved'}
                else:
                    comparisons['pairs'][f'K{k}'][pair_key][metric] = {'available': True, **paired_difference_ci(a, b, draws)}

    best = max(('bm25', 'colbert'), key=lambda arm: metrics['arms'][arm]['K3']['complete_support']['mean'])
    decision_diff = comparisons['pairs']['K3'][f'disco-{best}']['complete_support']
    delta = metrics['arms']['disco']['K3']['complete_support']['mean'] - metrics['arms'][best]['K3']['complete_support']['mean']
    ci = decision_diff['ci95']
    no_gain = delta < 0.05 or ci[0] <= 0 <= ci[1]
    comparisons['decision_K3'] = {
        'primary_metric': 'complete_support', 'best_comparator': best,
        'disco_minus_best': delta, 'ci95': ci,
        'conclusion': 'no practical gain from first-stage coverage' if no_gain else 'practical gain supported',
    }
    mismatches = []
    bm25_words = {k: metrics['arms']['bm25'][f'K{k}']['evidence_words']['mean'] for k in (2, 3, 5)}
    for arm in ('colbert', 'disco'):
        for k in (2, 3, 5):
            value = metrics['arms'][arm][f'K{k}']['evidence_words']['mean']
            if abs(value - bm25_words[k]) / bm25_words[k] > 0.15:
                mismatches.append({'arm': arm, 'k': k, 'relative_difference': abs(value - bm25_words[k]) / bm25_words[k]})
    metrics['budget_mismatch_flags'] = mismatches
    metrics['gold_sha256'] = gold_hash
    for output, value in (('metrics.json', metrics), ('comparisons.json', comparisons)):
        path = ROOT / 'outputs' / output
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({'scored': True, 'decision': comparisons['decision_K3'], 'budget_mismatches': len(mismatches)}))


if __name__ == '__main__':
    score()
