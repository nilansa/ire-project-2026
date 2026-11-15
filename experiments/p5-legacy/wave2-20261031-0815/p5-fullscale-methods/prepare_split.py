from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PILOT = Path('/Users/nilan/Documents/ire-p5-minimal-evidence/pilot')
OUT = Path(__file__).resolve().parent
DATA = PILOT / 'data/hotpot_validation.parquet'
SEED = 20261031

def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def norm(s): return ' '.join(str(s).split()).casefold()
def eligible(row):
    try:
        ctx=row['context']; titles=ctx['title']; groups=ctx['sentences']
        gold={norm(x) for x in row['supporting_facts']['title']}
        return (row['type'] in {'bridge','comparison'} and len(titles)==10 and len(groups)==10 and len(gold)==2 and sum(norm(t) in gold for t in titles)==2)
    except (KeyError,TypeError): return False

assert not any((OUT/n).exists() for n in ['SEQ_HOLDOUT.ids.txt','SEQ_HOLDOUT.ids.sha256','dev2.parquet','split_manifest.json'])
pilot=json.loads((PILOT/'results.json').read_text())
raw_sha=sha_file(DATA)
assert raw_sha==pilot['data']['sha256'], 'source hash differs from pilot record'
rows=pq.read_table(DATA).to_pylist()
elig=[r for r in rows if eligible(r)]
assert len(elig)==7342, f'eligible population changed: {len(elig)}'
sel=np.random.default_rng(SEED).choice(len(elig),size=600,replace=False)
sampled=[elig[int(i)] for i in sel]
sampled.sort(key=lambda r:(hashlib.sha256(str(r['id']).encode()).hexdigest(),str(r['id'])))
pilot_ids=[str(x) for x in pilot['sampling']['sampled_qids_in_hash_order']]
assert [str(r['id']) for r in sampled]==pilot_ids, 'pilot exclusion set mismatch'
excluded=set(pilot_ids)
remaining=[r for r in elig if str(r['id']) not in excluded]
assert len(remaining)==6742
remaining.sort(key=lambda r:(hashlib.sha256(str(r['id']).encode()).hexdigest(),str(r['id'])))
holdout=remaining[:2000]
holdout_ids=[str(r['id']) for r in holdout]
ids_bytes=(''.join(q+'\n' for q in holdout_ids)).encode()
ids_path=OUT/'SEQ_HOLDOUT.ids.txt'
ids_path.write_bytes(ids_bytes)
ids_hash=hashlib.sha256(ids_bytes).hexdigest()
(OUT/'SEQ_HOLDOUT.ids.sha256').write_text(ids_hash+'  SEQ_HOLDOUT.ids.txt\n')
# Holdout rows are dropped here; only DEV2 rows are materialized for evaluation.
dev2=remaining[2000:]
assert len(dev2)==4742 and not (set(holdout_ids)&{str(r['id']) for r in dev2})
pq.write_table(pa.Table.from_pylist(dev2),OUT/'dev2.parquet',compression='zstd')
manifest={
 'source_url':pilot['data']['source_url'],'source_sha256':raw_sha,'source_rows':len(rows),
 'eligible_rows':len(elig),'eligibility_exclusions':len(rows)-len(elig),
 'pilot_excluded_n':len(excluded),'pilot_seed':SEED,'pilot_ids_sha256':hashlib.sha256(('\n'.join(sorted(excluded))+'\n').encode()).hexdigest(),
 'remaining_n':len(remaining),'SEQ_HOLDOUT_n':len(holdout_ids),'SEQ_HOLDOUT_ids_sha256':ids_hash,
 'SEQ_HOLDOUT_order':'ascending sha256(UTF-8 question_id), tie-break question_id; first 2000',
 'DEV2_n':len(dev2),'DEV2_ids_sha256':hashlib.sha256((''.join(str(r['id'])+'\n' for r in dev2)).encode()).hexdigest(),
 'dev2_parquet_sha256':sha_file(OUT/'dev2.parquet'),'dev2_parquet_bytes':(OUT/'dev2.parquet').stat().st_size,
 'holdout_evaluated':False}
(OUT/'split_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['source_sha256','eligible_rows','pilot_excluded_n','remaining_n','SEQ_HOLDOUT_n','SEQ_HOLDOUT_ids_sha256','DEV2_n','dev2_parquet_bytes']}))
