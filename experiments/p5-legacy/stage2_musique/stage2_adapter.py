import json
from pathlib import Path

STAGE1 = Path('/Users/nilan/Documents/ire-p5-minimal-evidence/stage1_musique')
STAGE2 = Path(__file__).resolve().parent
DISCO = STAGE2 / 'DISCo'
DATASET = DISCO / 'data' / 'musique'
DOC_ID_MAP = STAGE2 / 'doc_id_map.json'
QID_ORDER = STAGE2 / 'qid_order.json'
QREL_HEADER = 'query-id\tcorpus-id\tscore\n'


def load_frozen_inputs(source=STAGE1):
    corpus = [json.loads(line) for line in (source / 'corpus.jsonl').open(encoding='utf-8')]
    questions = json.loads((source / 'sample_questions.json').read_text(encoding='utf-8'))
    if len(corpus) != 21100 or len({row['doc_id'] for row in corpus}) != 21100:
        raise ValueError('frozen corpus must contain 21,100 unique doc_id values')
    if len(questions) != 150 or len({row['id'] for row in questions}) != 150:
        raise ValueError('frozen question sample must contain 150 unique qids')
    return corpus, questions


def write_beir_dataset(corpus, questions, dataset_dir, map_path, qids_path):
    dataset_dir, map_path, qids_path = map(Path, (dataset_dir, map_path, qids_path))
    if dataset_dir.exists() or map_path.exists() or qids_path.exists():
        raise FileExistsError('refusing to overwrite an existing adapter artifact')
    dataset_dir.mkdir(parents=True)
    (dataset_dir / 'qrels').mkdir()
    with (dataset_dir / 'corpus.jsonl').open('x', encoding='utf-8') as f:
        for row in corpus:
            f.write(json.dumps({'_id': str(row['doc_id']), 'title': ' '.join(row['title'].split()), 'text': ' '.join(row['text'].split())}, ensure_ascii=False) + '\n')
    with (dataset_dir / 'queries.jsonl').open('x', encoding='utf-8') as f:
        for row in questions:
            f.write(json.dumps({'_id': str(row['id']), 'text': row['question']}, ensure_ascii=False) + '\n')
    with (dataset_dir / 'qrels' / 'test.tsv').open('x', encoding='utf-8') as f:
        f.write(QREL_HEADER)
    map_path.write_text(json.dumps([str(row['doc_id']) for row in corpus], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    qids_path.write_text(json.dumps([str(row['id']) for row in questions], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert_qrel_blind(dataset_dir / 'qrels' / 'test.tsv')


def assert_qrel_blind(qrels_path):
    lines = Path(qrels_path).read_text(encoding='utf-8').splitlines()
    if not lines or lines[0] != QREL_HEADER.rstrip('\n'):
        raise AssertionError('qrels/test.tsv must have only the standard header')
    rows = [line for line in lines[1:] if line.strip()]
    if rows:
        raise AssertionError(f'expected 0 qrel rows, found {len(rows)}')
    return 0


def map_internal_ids(ids, doc_ids):
    out = []
    for value in ids:
        idx = int(value)
        if idx < 0 or idx >= len(doc_ids):
            raise IndexError(f'internal doc id outside adapter map: {idx}')
        out.append(str(doc_ids[idx]))
    return out


def evidence_word_count(title, text):
    return len((title + ' ' + text).split())


def prepare_frozen_adapter():
    corpus, questions = load_frozen_inputs()
    write_beir_dataset(corpus, questions, DATASET, DOC_ID_MAP, QID_ORDER)
    print(json.dumps({'docs': len(corpus), 'questions': len(questions), 'qrel_rows': 0, 'dataset': str(DATASET)}))


if __name__ == '__main__':
    prepare_frozen_adapter()
