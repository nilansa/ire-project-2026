import datetime as dt
import hashlib
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from stage2_adapter import DATASET, DOC_ID_MAP, QID_ORDER, STAGE1, STAGE2, assert_qrel_blind, evidence_word_count, load_frozen_inputs

ROOT = STAGE2
DISCO = ROOT / 'DISCo'
GOLD = STAGE1 / 'gold_HIDDEN.json'
GOLD_SHA256 = '084e5502ab3d61a368968c61c99e2949969e6a9bd40583fd9d35d99ad650185e'
MANIFEST_SHA256 = '9bbea3f6a3eb014b6c401715980e80e3f777da79f9a8bbca79e6387ec39e0832'
QRELS = DATASET / 'qrels' / 'test.tsv'
PYTHON = Path(sys.executable)
ENV = os.environ.copy()
ENV.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'})


class RunFailure(RuntimeError):
    pass


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def append_status(title, detail):
    path = ROOT / 'STATUS.md'
    with path.open('a', encoding='utf-8') as f:
        f.write(f'\n## {now_iso()} — {title}\n\n{detail}\n')
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def verify_frozen_non_gold():
    manifest = STAGE1 / 'MANIFEST_FROZEN.sha256'
    results = STAGE1 / 'RESULTS.sha256'
    if sha256(manifest) != MANIFEST_SHA256:
        raise RunFailure('MANIFEST_FROZEN.sha256 digest mismatch')
    wanted = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            wanted[rel] = digest
    for rel in ('corpus.jsonl', 'sample_questions.json'):
        if sha256(STAGE1 / rel) != wanted[rel]:
            raise RunFailure(f'frozen input digest mismatch: {rel}')
    result_wanted = {}
    for line in results.read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            result_wanted[rel] = digest
    for rel in ('predictions_top5.json', 'retrieve_log.json'):
        if sha256(STAGE1 / rel) != result_wanted[rel]:
            raise RunFailure(f'frozen result digest mismatch: {rel}')


def prepare_bm25_outputs():
    output_root = ROOT / 'outputs'
    for name in ('predictions', 'evidence_words', 'latency', 'runs', 'logs', 'checkpoints'):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    frozen_predictions = json.loads((STAGE1 / 'predictions_top5.json').read_text(encoding='utf-8'))
    frozen_latency = json.loads((STAGE1 / 'retrieve_log.json').read_text(encoding='utf-8'))
    corpus, questions = load_frozen_inputs()
    docs = {str(row['doc_id']): row for row in corpus}
    qids = [str(row['id']) for row in questions]
    if list(frozen_predictions) != qids:
        raise RunFailure('frozen BM25 qid order differs from question sample')
    for qid in qids:
        for item in frozen_predictions[qid][:5]:
            doc = docs[str(item['doc_id'])]
            if int(item['words']) != evidence_word_count(doc['title'], doc['text']):
                raise RunFailure(f'Stage-1 evidence word mismatch for {qid}/{item["doc_id"]}')
    for k in (2, 3, 5):
        predictions = {qid: [str(item['doc_id']) for item in frozen_predictions[qid][:k]] for qid in qids}
        words = {qid: sum(int(item['words']) for item in frozen_predictions[qid][:k]) for qid in qids}
        if any(len(predictions[qid]) != k for qid in qids):
            raise RunFailure(f'frozen BM25 returned fewer than {k} documents')
        write_json_new(output_root / 'predictions' / f'bm25_k{k}.json', predictions)
        write_json_new(output_root / 'evidence_words' / f'bm25_k{k}.json', words)
        write_json_new(output_root / 'latency' / f'bm25_k{k}.json', {
            'per_query_available': False,
            'mean_s': frozen_latency['latency_mean_s'],
            'p95_s': frozen_latency['latency_p95_s'],
            'source': 'stage1_musique/retrieve_log.json; BM25 not rerun',
        })
    write_json_new(output_root / 'runs' / 'bm25.json', {
        'method': 'frozen_stage1_bm25', 'n_queries': len(qids), 'n_docs': len(corpus),
        'index_seconds': frozen_latency['index_s'], 'mean_query_seconds': frozen_latency['latency_mean_s'],
        'p95_query_seconds': frozen_latency['latency_p95_s'], 'per_query_latencies_saved': False,
        'verifier_calls': 0,
    })


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def command_log(name, argv, cwd):
    path = ROOT / 'RUN_COMMANDS.log'
    with path.open('a', encoding='utf-8') as f:
        f.write(f'[{now_iso()}] {name}\n')
        f.write(f'cwd={cwd}\n')
        f.write(shlex.join([str(x) for x in argv]) + '\n\n')
        f.flush()
        os.fsync(f.fileno())


def log_tail(path, n=30):
    try:
        return '\n'.join(Path(path).read_text(encoding='utf-8', errors='replace').splitlines()[-n:])
    except Exception:
        return '(log unavailable)'


def run_command(name, argv, cwd, timeout=None, qrel_check=False):
    if qrel_check and assert_qrel_blind(QRELS) != 0:
        raise RunFailure(f'{name}: qrels must contain zero rows')
    command_log(name, argv, cwd)
    log_path = ROOT / 'outputs' / 'logs' / f'{name}.log'
    start_iso = now_iso()
    start = time.monotonic()
    with log_path.open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(
            [str(x) for x in argv], cwd=str(cwd), env=ENV,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            elapsed = time.monotonic() - start
            append_jsonl(ROOT / 'outputs' / 'logs' / 'steps.jsonl', {
                'step': name, 'start': start_iso, 'end': now_iso(), 'elapsed_s': elapsed,
                'exit_code': 'timeout', 'timeout_s': timeout, 'log': str(log_path),
            })
            raise RunFailure(f'{name} timed out after {elapsed:.2f}s (limit {timeout}s)')
    elapsed = time.monotonic() - start
    append_jsonl(ROOT / 'outputs' / 'logs' / 'steps.jsonl', {
        'step': name, 'start': start_iso, 'end': now_iso(), 'elapsed_s': elapsed,
        'exit_code': code, 'log': str(log_path),
    })
    if code != 0:
        raise RunFailure(f'{name} exited {code} after {elapsed:.2f}s; log tail:\n{log_tail(log_path)}')
    return elapsed, log_path


def assert_new_index_space():
    paths = [
        DISCO / 'colbert_beir_expts',
        DISCO / 'experiments' / 'musique',
        DISCO / 'docid_to_batchinfo' / 'docid_to_batchinfo_musique.json',
        DISCO / 'pickles' / 'results' / 'BERT',
    ]
    for path in paths:
        if path.exists() and (path.is_file() or any(path.iterdir())):
            raise RunFailure(f'refusing to overwrite pre-existing index artifact: {path}')


def write_prediction_manifest():
    files = [ROOT / 'outputs' / 'predictions' / f'{arm}_k{k}.json' for arm in ('bm25', 'colbert', 'disco') for k in (2, 3, 5)]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise RunFailure('cannot hash incomplete predictions: ' + ', '.join(missing))
    manifest = ROOT / 'PREDICTIONS.sha256'
    with manifest.open('x', encoding='utf-8') as f:
        for path in sorted(files):
            f.write(f'{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n')
    run_command('verify_predictions_hashes', ['shasum', '-a', '256', '-c', str(manifest)], ROOT, qrel_check=False)


def build_indexes_and_retrieve():
    global_start = dt.datetime.now().astimezone()
    append_status('Detached worker started', f'PID `{os.getpid()}`; start `{global_start.isoformat(timespec="seconds")}`. CPU-only; no gold reads during retrieval.')
    verify_frozen_non_gold()
    if assert_qrel_blind(QRELS) != 0:
        raise RunFailure('adapter qrels are not empty')
    gold_mode = stat.S_IMODE(GOLD.stat().st_mode)
    if gold_mode != 0o600:
        raise RunFailure(f'expected gold mode 600 before retrieval, got {gold_mode:04o}')
    assert_new_index_space()
    expected = global_start + dt.timedelta(hours=6)
    append_status('Expected finish', f'Current estimate `{expected.isoformat(timespec="seconds")}` (six hours after launch); refine after the PLAID index timing. The F-CPU index deadline is 5,400 seconds after launch.')
    prepare_bm25_outputs()

    os.chmod(GOLD, 0)
    if stat.S_IMODE(GOLD.stat().st_mode) != 0:
        raise RunFailure('failed to set gold mode 000')
    append_status('Retrieval phase locked', 'gold_HIDDEN.json mode is 000; retrieval worker does not open it.')

    index_records = ROOT / 'outputs' / 'index_builds.jsonl'
    baseline_cmd = [
        PYTHON, '-m', 'src.retrievalmethods',
        'data.dataset_name=musique', 'data.loader_type=beir', 'data.query_type=forum',
        'method=baseline', 'overwrite_index=True', 'index=True', 'augment=False',
        'embedder.mode=disk', 'embedder.mv_type=colbertv2-plaid', 'colbert.nbits=2',
    ]
    assert_qrel_blind(QRELS)
    baseline_s, _ = run_command('plaid_baseline_index', baseline_cmd, DISCO, timeout=5400, qrel_check=True)
    base_index = DISCO / 'colbert_beir_expts' / 'musique' / 'BERT' / 'colbertv2-plaid' / 'norm' / 'indexes' / 'nbits=2.noaug'
    if not base_index.exists():
        raise RunFailure('baseline PLAID index command exited successfully but expected index path is missing')
    append_jsonl(index_records, {'arm': 'colbert_baseline', 'elapsed_s': baseline_s, 'index': str(base_index), 'finished': now_iso()})
    append_status('F-CPU PLAID index passed', f'21,100-document baseline PLAID index completed in `{baseline_s:.2f}s` (limit 5,400s).')

    map_cmd = [PYTHON, '../src/calculate_docid_to_batch_info.py', 'musique']
    assert_qrel_blind(QRELS)
    run_command('docid_batch_map', map_cmd, DISCO / 'ColBERT', qrel_check=True)
    mapping_path = DISCO / 'docid_to_batchinfo' / 'docid_to_batchinfo_musique.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    if len(mapping) != 21100:
        raise RunFailure(f'docid-to-batch map has {len(mapping)} entries, expected 21,100')

    for rh in range(8):
        cmd = [
            PYTHON, '-m', 'src.retrievalmethods',
            'data.dataset_name=musique', 'data.loader_type=beir', 'data.query_type=forum',
            'method=disco', 'overwrite_index=True', 'index=True', 'augment=True', 'dbl_norm=True',
            'generate_new_rh=True', f'rh_num={rh}', 'num_rh_augment=8',
            'embedder.mode=disk', 'embedder.mv_type=colbertv2-plaid', 'colbert.nbits=2',
            'colbert_topk=1', 'disco.rerank_internal=True', 'disco.rerank_external=True',
        ]
        assert_qrel_blind(QRELS)
        elapsed, _ = run_command(f'disco_index_rh{rh}', cmd, DISCO, qrel_check=True)
        index = DISCO / 'colbert_beir_expts' / 'musique' / 'BERT' / 'colbertv2-plaid' / 'dblnorm' / 'indexes' / f'nbits=2.aug8.{rh}'
        if not index.exists():
            raise RunFailure(f'DISCo random-hyperplane index {rh} missing after build')
        append_jsonl(index_records, {'arm': 'disco', 'rh': rh, 'elapsed_s': elapsed, 'index': str(index), 'finished': now_iso()})
        append_status(f'DISCo index {rh}/8 complete', f'Elapsed `{elapsed:.2f}s`; path `{index}`.')

    for name, arg in (('colbert_queries', 'baseline'), ('disco_k2', 'disco2'), ('disco_k3', 'disco3'), ('disco_k5', 'disco5')):
        assert_qrel_blind(QRELS)
        elapsed, _ = run_command(name, [PYTHON, ROOT / 'stage2_runtime.py', arg], ROOT, qrel_check=True)
        append_status(f'{name} complete', f'Elapsed `{elapsed:.2f}s`; qrels had zero rows.')
        if name == 'disco_k2':
            append_status('F-CPU passed', 'Baseline PLAID index completed within 5,400 seconds and DISCo K=2 completed on all 150 queries without error (at least five required).')

    write_prediction_manifest()
    append_status('Predictions hashed', 'All nine prediction files are listed in PREDICTIONS.sha256 and pass `shasum -a 256 -c`; scoring has not started.')


def restore_gold():
    os.chmod(GOLD, 0o600)
    mode = stat.S_IMODE(GOLD.stat().st_mode)
    digest = sha256(GOLD)
    if mode != 0o600 or digest != GOLD_SHA256:
        raise RunFailure(f'gold restore verification failed: mode={mode:04o} sha256={digest}')
    path = ROOT / 'GOLD_RESTORED.sha256'
    with path.open('x', encoding='utf-8') as f:
        f.write(f'{digest}  gold_HIDDEN.json (mode {mode:04o}; content verified unchanged)\n')
    append_status('Gold restored and verified', f'Mode `{mode:04o}`; SHA-256 `{digest}` matches the frozen manifest.')


def main():
    if not (ROOT / 'STATUS.md').is_file():
        raise RunFailure('STATUS.md must be initialized with the tmux session and launcher PID before worker start')
    locked = False
    failed = None
    try:
        build_indexes_and_retrieve()
        locked = True
    except BaseException as exc:
        failed = exc
        mode = stat.S_IMODE(GOLD.stat().st_mode)
        locked = mode == 0
    finally:
        if locked:
            try:
                restore_gold()
            except BaseException as restore_exc:
                if failed is None:
                    failed = restore_exc
                else:
                    failed = RunFailure(f'{failed}; additionally {restore_exc}')
    if failed is not None:
        append_status('BLOCKED_CPU', f'Error: `{type(failed).__name__}: {failed}`. See `outputs/logs/` for exact command logs. No scoring or retry was performed.')
        raise SystemExit(1)

    score_cmd = [PYTHON, ROOT / 'score_stage2.py']
    try:
        run_command('score_once', score_cmd, ROOT, qrel_check=False)
    except BaseException as exc:
        append_status('SCORING_FAILED', f'`{type(exc).__name__}: {exc}`. Predictions remain hashed; no scoring rerun was performed.')
        raise SystemExit(1)
    append_status('Completed', 'Scoring ran once after prediction hash verification and gold mode/hash restoration. Results: `outputs/metrics.json` and `outputs/comparisons.json`.')


if __name__ == '__main__':
    main()
