#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet

BASE = Path(__file__).resolve().parent
DATA = Path('/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/data/hotpot_validation.parquet')
PILOT_CACHE = Path('/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/.hf-cache')
ENCODER_ID = 'sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CROSS_ID = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
SEED = 20261031
EXPECTED_DATA_SHA = 'c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6'
EXPECTED_QID_SHA = '6ba3cab6f2ad2208e044906eee159c43b45c3cb14511a0a5be6646d8cb860c37'
ENCODER_MAX_LENGTH = 256
MAX_CROSS_BYTES = 150_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(directory: Path) -> tuple[int, list[dict[str, Any]]]:
    files = []
    for p in sorted(directory.rglob('*')):
        if p.is_file():
            files.append({'path': str(p.relative_to(directory)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)})
    return sum(x['bytes'] for x in files), files


def normalize_title(value: str) -> str:
    return ' '.join(value.split()).casefold()


def prepare(row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row['id'])
    question = str(row['question'])
    kind = str(row['type'])
    if kind not in {'bridge', 'comparison'}:
        raise ValueError(f'Unexpected question type for {qid}: {kind}')
    context = row['context']
    titles = [str(x) for x in context['title']]
    sentence_groups = context['sentences']
    if len(titles) < 2 or len(titles) != len(sentence_groups):
        raise ValueError(f'Invalid candidate paragraph count for {qid}: {len(titles)}')
    paragraphs = []
    for title, sentences in zip(titles, sentence_groups):
        body = ' '.join(str(s).strip() for s in sentences)
        paragraphs.append((title.strip() + ' ' + body).strip())
    facts = row['supporting_facts']
    gold_titles = {normalize_title(str(t)) for t in facts['title']}
    if len(gold_titles) != 2:
        raise ValueError(f'Expected two unique supporting titles for {qid}')
    gold = {i for i, title in enumerate(titles) if normalize_title(title) in gold_titles}
    if len(gold) != 2:
        raise ValueError(f'Gold titles do not map to two candidates for {qid}')
    answer = str(row['answer'])
    if not answer.strip():
        raise ValueError(f'Empty answer for {qid}')
    return {'qid': qid, 'type': kind, 'question': question, 'answer': answer,
            'paragraphs': paragraphs, 'gold': gold}


def load_encoder(torch: Any, transformers: Any) -> tuple[Any, Any, dict[str, Any]]:
    snapshot = PILOT_CACHE / 'models--sentence-transformers--all-MiniLM-L6-v2' / 'snapshots' / ENCODER_REVISION
    if not snapshot.is_dir():
        raise FileNotFoundError(f'Pilot encoder snapshot is missing: {snapshot}')
    total, files = file_manifest(snapshot)
    if total > 150_000_000:
        raise RuntimeError(f'Encoder snapshot exceeds 150 MB: {total}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = transformers.AutoModel.from_pretrained(str(snapshot), local_files_only=True).to('cpu').eval()
    return tokenizer, model, {'id': ENCODER_ID, 'revision': ENCODER_REVISION,
                               'source': str(snapshot), 'device': 'cpu',
                               'max_length': ENCODER_MAX_LENGTH,
                               'model_file_bytes': total, 'files': files}


def encode(texts: list[str], tokenizer: Any, model: Any, torch: Any) -> np.ndarray:
    batch = tokenizer(texts, padding=True, truncation=True, max_length=ENCODER_MAX_LENGTH,
                      return_tensors='pt')
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return vectors.cpu().numpy()


def mmr_order(query: np.ndarray, candidates: np.ndarray) -> list[int]:
    qsim = candidates @ query
    selected: list[int] = []
    while len(selected) < 2:
        best, best_score = -1, -float('inf')
        for i in range(len(candidates)):
            if i in selected:
                continue
            score = float(qsim[i]) if not selected else 0.5 * float(qsim[i]) - 0.5 * float(np.max(candidates[i] @ candidates[selected].T))
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
    return selected


def load_cross_encoder(torch: Any, hub: Any, transformers: Any) -> tuple[Any, Any, dict[str, Any]]:
    cache_dir = BASE / '.hf-cache'
    local_root = cache_dir / 'models--cross-encoder--ms-marco-MiniLM-L-6-v2' / 'snapshots'
    snapshot = None
    if local_root.is_dir():
        snapshots = sorted(p for p in local_root.iterdir() if p.is_dir())
        if snapshots:
            snapshot = snapshots[-1]
    if snapshot is None:
        api = hub.HfApi(token=False)
        info = api.model_info(CROSS_ID, revision='main', files_metadata=True)
        sizes = {}
        for item in info.siblings:
            name = item.rfilename
            size = getattr(item, 'size', None)
            if size is None and getattr(item, 'lfs', None) is not None:
                size = getattr(item.lfs, 'size', None)
            if size is not None:
                sizes[name] = int(size)
        weight = 'model.safetensors' if 'model.safetensors' in sizes else 'pytorch_model.bin'
        candidates = ['config.json', weight, 'tokenizer.json', 'tokenizer_config.json',
                      'special_tokens_map.json', 'vocab.txt', 'merges.txt', 'added_tokens.json']
        allowed = [name for name in candidates if name in sizes]
        if weight not in allowed:
            raise RuntimeError('No supported cross-encoder weight file in model metadata')
        expected_bytes = sum(sizes[name] for name in allowed)
        if expected_bytes > MAX_CROSS_BYTES:
            raise RuntimeError(f'Cross-encoder files exceed 150 MB: {expected_bytes}')
        snapshot = Path(hub.snapshot_download(
            repo_id=CROSS_ID, revision=info.sha, cache_dir=str(cache_dir),
            allow_patterns=allowed, token=False,
        ))
    total, files = file_manifest(snapshot)
    if total > MAX_CROSS_BYTES:
        raise RuntimeError(f'Cross-encoder files exceed 150 MB: {total}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(snapshot), local_files_only=True).to('cpu').eval()
    return tokenizer, model, {'id': CROSS_ID, 'revision': snapshot.name,
                              'source': str(snapshot), 'device': 'cpu',
                              'model_file_bytes': total, 'files': files,
                              'download_cap_bytes': MAX_CROSS_BYTES}


def score_pairs(pairs: list[tuple[int, int, str, str]], tokenizer: Any,
                model: Any, torch: Any) -> tuple[dict[tuple[int, int], float], int]:
    out: dict[tuple[int, int], float] = {}
    truncated = 0
    for start in range(0, len(pairs), 24):
        batch = pairs[start:start + 24]
        encoded = tokenizer([p[2] for p in batch], [p[3] for p in batch],
                            padding=True, truncation=True, max_length=512,
                            return_tensors='pt')
        if int(encoded['input_ids'].shape[1]) >= 512:
            truncated += sum(1 for row in encoded['attention_mask'] if int(row.sum()) >= 512)
        with torch.inference_mode():
            logits = model(**encoded).logits.reshape(-1).cpu().tolist()
        for item, score in zip(batch, logits):
            out[(item[0], item[1])] = float(score)
    return out, truncated


def answer_present(answer: str, selected_text: str) -> int:
    norm = lambda s: ' '.join(s.split()).casefold()
    return int(norm(answer) in norm(selected_text))


def auc(scores: list[float], labels: list[int]) -> float | None:
    a = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    pos, neg = a[y == 1], a[y == 0]
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def choose_threshold(scores: list[float], labels: list[int]) -> dict[str, Any]:
    pairs = sorted(set(float(s) for s in scores))
    for threshold in pairs:
        pred = [s >= threshold for s in scores]
        n = sum(pred)
        if n:
            precision = sum(int(p and y) for p, y in zip(pred, labels)) / n
            if precision >= 0.90:
                return {'threshold': threshold, 'dev_precision': precision,
                        'dev_predicted_sufficient_n': n, 'dev_n': len(labels)}
    return {'threshold': None, 'dev_precision': None,
            'dev_predicted_sufficient_n': 0, 'dev_n': len(labels)}


def metric_group(scores: list[float], labels: list[int], threshold: float | None) -> dict[str, Any]:
    positives = sum(labels)
    negatives = len(labels) - positives
    predicted = [threshold is not None and s >= threshold for s in scores]
    fp = sum(int(p and y == 0) for p, y in zip(predicted, labels))
    tp = sum(int(p and y == 1) for p, y in zip(predicted, labels))
    return {'n': len(labels), 'full_support_n': positives, 'insufficient_n': negatives,
            'auroc': auc(scores, labels), 'threshold': threshold,
            'predicted_sufficient_n': sum(predicted), 'false_sufficient_n': fp,
            'holdout_precision': tp / (tp + fp) if tp + fp else None,
            'false_sufficient_rate': (fp / negatives if negatives else None) if threshold is not None else None}


def build_report(result: dict[str, Any]) -> str:
    lines = [
        '# Sufficiency detector results', '',
        f"- Data SHA256: `{result['data']['sha256']}`; sampled 600 (200 dev / 400 holdout), NumPy seed {SEED}.",
        f"- Candidate pool sizes: {result['sampling']['candidate_count_distribution']} (all sampled rows retained).",
        f"- Encoder: `{result['models']['encoder']['id']}` revision `{result['models']['encoder']['revision']}` (CPU).",
        f"- Cross-encoder: `{result['models']['cross_encoder'].get('id', 'unavailable')}`; status {result['models']['cross_encoder']['status']}.",
        f"- NLI: {result['models']['nli']['status']}; label `ORACLE-ANSWER-DIAGNOSTIC` if run.", '',
        '## Holdout AUROC and false-sufficient rate', '',
        'FSR thresholds were selected on dev for at least 90% precision. FSR is false positives / actually insufficient.', '',
        '| Selector | Detector | Overall AUROC | Overall FSR | Bridge AUROC / FSR | Comparison AUROC / FSR |',
        '|---|---|---:|---:|---:|---:|'
    ]
    for selector, detectors in result['metrics'].items():
        for detector, data in detectors.items():
            groups = data['holdout']
            def fmt(group: dict[str, Any]) -> str:
                a = group['auroc']; f = group['false_sufficient_rate']
                return f"{a:.3f}" if a is not None else 'NA', f"{f:.3f}" if f is not None else 'NA'
            oa, of = fmt(groups['overall']); ba, bf = fmt(groups['bridge']); ca, cf = fmt(groups['comparison'])
            th = data['threshold_selection']
            label = (f"{detector} (dev t={th['threshold']}, precision={th['dev_precision']}, "
                     f"predicted={th['dev_predicted_sufficient_n']}/{th['dev_n']})")
            lines.append(f'| {selector} | {label} | {oa} | {of} | {ba} / {bf} | {ca} / {cf} |')
    lines += ['', '## S1 gate', '']
    s1 = result['gate_S1']
    if s1['status'] == 'evaluated':
        lines.append(f"**{s1['result']}** — best non-leaking holdout AUROC {s1['best_auroc']:.3f} (`{s1['best_detector']}`); cutoff <= 0.75.")
    else:
        lines.append(f"**NOT EVALUABLE** — {s1['reason']}")
    lines += ['', '## Scope notes', '']
    lines += [f'- NLI: {result["models"]["nli"].get("reason", result["models"]["nli"]["status"])}.',
              f'- Cross-encoder status: {result["models"]["cross_encoder"].get("reason", result["models"]["cross_encoder"]["status"])}.',
              f'- Elapsed wall time: {result["runtime"]["elapsed_seconds"]:.1f} seconds.']
    return '\n'.join(lines) + '\n'


def main() -> None:
    for output in (BASE / 'results.json', BASE / 'results.md'):
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite existing artifact: {output}')
    if sha256_file(DATA) != EXPECTED_DATA_SHA:
        raise ValueError('Pilot dataset SHA256 mismatch')
    started = time.perf_counter()
    rows = parquet.read_table(DATA).to_pylist()
    if len(rows) < 600:
        raise ValueError(f'Only {len(rows)} source rows; need 600')
    chosen = np.random.default_rng(SEED).choice(len(rows), size=600, replace=False)
    sampled = [rows[int(i)] for i in chosen]
    if len({str(r['id']) for r in sampled}) != 600:
        raise ValueError('Sample does not contain 600 unique qids')
    sampled.sort(key=lambda r: (hashlib.sha256(str(r['id']).encode('utf-8')).hexdigest(), str(r['id'])))
    qids = [str(r['id']) for r in sampled]
    qid_sha = hashlib.sha256(json.dumps(qids, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
    if qid_sha != EXPECTED_QID_SHA:
        raise ValueError(f'Sampled qids differ from pilot manifest: {qid_sha}')
    examples = [prepare(r) for r in sampled]
    split = ['dev'] * 200 + ['holdout'] * 400
    for item, part in zip(examples, split):
        item['split'] = part

    import torch
    import transformers
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    encoder_tok, encoder, encoder_info = load_encoder(torch, transformers)
    selections: list[dict[str, Any]] = []
    pair_keys: set[tuple[int, int]] = set()
    pair_list: list[tuple[int, int, str, str]] = []
    for q_index, item in enumerate(examples):
        vectors = encode([item['question'], *item['paragraphs']], encoder_tok, encoder, torch)
        qvec, pvec = vectors[0], vectors[1:]
        dense = np.argsort(-(pvec @ qvec), kind='stable').astype(int).tolist()[:2]
        mmr = mmr_order(qvec, pvec)
        selected = {'dense': dense, 'mmr': mmr}
        selections.append(selected)
        for method in ('dense', 'mmr'):
            for candidate in selected[method]:
                key = (q_index, candidate)
                if key not in pair_keys:
                    pair_keys.add(key)
                    pair_list.append((q_index, candidate, item['question'], item['paragraphs'][candidate]))

    cross_info: dict[str, Any]
    pair_scores: dict[tuple[int, int], float] = {}
    cross_truncated = 0
    cross_tok = cross_model = None
    try:
        import huggingface_hub
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        cross_tok, cross_model, cross_info = load_cross_encoder(torch, huggingface_hub, transformers)
        cross_info['status'] = 'loaded'
        pair_scores, cross_truncated = score_pairs(pair_list, cross_tok, cross_model, torch)
    except Exception as exc:
        snapshots = sorted((BASE / '.hf-cache' / 'models--cross-encoder--ms-marco-MiniLM-L-6-v2' / 'snapshots').glob('*'))
        partial = list((BASE / '.hf-cache' / 'models--cross-encoder--ms-marco-MiniLM-L-6-v2' / 'blobs').glob('*.incomplete'))
        cross_info = {'id': CROSS_ID, 'revision': snapshots[-1].name if snapshots else None,
                      'status': 'unavailable', 'download_attempted': True,
                      'incomplete_weight_bytes': sum(p.stat().st_size for p in partial),
                      'reason': f'Pinned checkpoint weight did not complete; no retry or model substitution. {type(exc).__name__}: {str(exc)[:250]}'}

    detector_values: dict[str, dict[str, list[float]]] = {}
    for method in ('dense', 'mmr'):
        vals: dict[str, list[float]] = {'answer_presence': []}
        if cross_info['status'] == 'loaded':
            vals.update({'cross_encoder_sum': [], 'cross_encoder_min': []})
        for q_index, item in enumerate(examples):
            picked = selections[q_index][method]
            text = ' '.join(item['paragraphs'][i] for i in picked)
            vals['answer_presence'].append(float(answer_present(item['answer'], text)))
            if cross_info['status'] == 'loaded':
                scores = [pair_scores[(q_index, i)] for i in picked]
                vals['cross_encoder_sum'].append(float(sum(scores)))
                vals['cross_encoder_min'].append(float(min(scores)))
        detector_values[method] = vals

    metrics: dict[str, Any] = {}
    for method, detectors in detector_values.items():
        metrics[method] = {}
        for detector, scores in detectors.items():
            dev_idx = [i for i, item in enumerate(examples) if item['split'] == 'dev']
            threshold = choose_threshold([scores[i] for i in dev_idx],
                                         [int(examples[i]['gold'] <= set(selections[i][method])) for i in dev_idx])
            cutoff = threshold['threshold']
            groups: dict[str, Any] = {}
            for group in ('overall', 'bridge', 'comparison'):
                idx = [i for i, item in enumerate(examples)
                       if item['split'] == 'holdout' and (group == 'overall' or item['type'] == group)]
                y = [int(examples[i]['gold'] <= set(selections[i][method])) for i in idx]
                s = [scores[i] for i in idx]
                groups[group] = metric_group(s, y, cutoff)
            metrics[method][detector] = {'threshold_selection': threshold, 'holdout': groups}

    if cross_info['status'] == 'loaded':
        candidates = [(method, detector, data['holdout']['overall']['auroc'])
                      for method, ds in metrics.items() for detector, data in ds.items()
                      if detector in {'cross_encoder_sum', 'cross_encoder_min'}
                      and data['holdout']['overall']['auroc'] is not None]
        if candidates:
            best = max(candidates, key=lambda x: x[2])
            s1 = {'status': 'evaluated', 'best_selector': best[0], 'best_detector': best[1],
                  'best_auroc': best[2], 'threshold': 0.75,
                  'result': 'PASS' if best[2] <= 0.75 else 'FAIL'}
        else:
            s1 = {'status': 'not_evaluable', 'reason': 'AUROC undefined for cross-encoder detectors'}
    else:
        s1 = {'status': 'not_evaluable', 'reason': 'Cross-encoder unavailable; no non-leaking detector AUROC'}

    nli_dir = BASE / '.hf-cache'
    nli_found = list(nli_dir.glob('models--*nli*')) if nli_dir.exists() else []
    nli = {'status': 'unavailable', 'reason': 'No cached NLI checkpoint was present; oracle-answer diagnostic skipped.'}
    if nli_found:
        nli['reason'] = 'A candidate cache directory exists, but this fixed run does not load unverified NLI artifacts.'
    types = Counter((item['split'], item['type']) for item in examples)
    runtime = {
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'cpu': platform.processor() or platform.machine(), 'logical_cpus': os.cpu_count(),
        'torch_threads': torch.get_num_threads(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'numpy': np.__version__,
        'pyarrow': parquet.__version__ if hasattr(parquet, '__version__') else __import__('pyarrow').__version__,
        'huggingface_hub': getattr(__import__('huggingface_hub'), '__version__', None) if cross_info['status'] == 'loaded' else None,
        'elapsed_seconds': time.perf_counter() - started,
    }
    result = {
        'task_id': 'ire-p5-suff-luna-002',
        'protocol': 'protocol.md',
        'data': {'path': str(DATA), 'sha256': EXPECTED_DATA_SHA, 'bytes': DATA.stat().st_size,
                 'rows': len(rows), 'source_url': 'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet'},
        'sampling': {'seed': SEED, 'sample_n': 600, 'dev_n': 200, 'holdout_n': 400,
                     'qid_ordered_sha256': qid_sha, 'same_as_pilot_protocol': True,
                     'candidate_count_distribution': {str(n): int(c) for n, c in sorted(Counter(len(x['paragraphs']) for x in examples).items())},
                     'type_counts': {part: {kind: int(types[(part, kind)]) for kind in ('bridge', 'comparison')}
                                     for part in ('dev', 'holdout')}},
        'models': {'encoder': encoder_info, 'cross_encoder': cross_info, 'nli': nli},
        'methods': {'dense': 'top-2 cosine on masked-mean pooled normalized MiniLM vectors',
                    'mmr': 'top-2 greedy MMR, lambda=0.5; stable candidate-order ties'},
        'ground_truth': 'both unique gold supporting-fact paragraphs selected',
        'threshold_rule': 'lowest dev threshold with >=0.90 precision and >=1 predicted positive',
        'fsr_definition': 'false positives / actually insufficient holdout examples',
        'metrics': metrics, 'gate_S1': s1,
        'nli_label': 'ORACLE-ANSWER-DIAGNOSTIC',
        'per_query': [
            {'qid': item['qid'], 'split': item['split'], 'type': item['type'],
             'candidate_n': len(item['paragraphs']),
             'full_support': {method: bool(item['gold'] <= set(selections[i][method])) for method in ('dense', 'mmr')},
             'selected_candidate_indices': selections[i],
             'detector_scores': {method: {det: values[i] for det, values in detector_values[method].items()}
                                 for method in ('dense', 'mmr')}}
            for i, item in enumerate(examples)
        ],
        'runtime': runtime,
        'cross_encoder_truncated_pairs_at_512': cross_truncated,
    }
    out_json, out_md = BASE / 'results.json', BASE / 'results.md'
    with out_json.open('x', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    with out_md.open('x', encoding='utf-8') as f:
        f.write(build_report(result))
    print(json.dumps({'results_json': str(out_json), 'results_md': str(out_md),
                      'sample_qid_sha256': qid_sha, 'cross_encoder': cross_info['status'],
                      's1': s1, 'elapsed_seconds': runtime['elapsed_seconds']}, sort_keys=True))


if __name__ == '__main__':
    main()
