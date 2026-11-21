"""Separate final evaluation; never called during selection, scoring or fitting."""
from pathlib import Path
import json,sys,hashlib,subprocess
import numpy as np
from metrics import metrics, paired

ARMS=['dense','dense10','coverage','bert_maxsim','rankmatched_0','rankmatched_1',
      'rankmatched_2','uniform','rerank_dense10','rerank_coverage']
QRELS_SHA='0d9458899c35efb08c2c3d6330995d0b76851222127e557a6f5b930c88f86261'

def verify_digest(path, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected: raise ValueError('input digest mismatch')

def validate_rankings(rankings,corpus):
    for a in ARMS:
        ids=rankings.get(a,[]); expected=10 if a.startswith('rerank_') else 100
        if len(ids)!=expected or len(set(ids))!=expected or not set(ids)<=corpus: raise ValueError('invalid ranking: '+a)

def evaluate_one(qid, rankings, gold, chosen, pool):
    if not gold or not set(chosen)<=set(pool): raise ValueError('missing qrels or invalid selected/pool mapping: '+qid)
    if any(a not in rankings for a in ARMS): raise ValueError('missing arm: '+qid)
    scores={a:metrics(rankings[a],gold) for a in ARMS}
    positives={str(d) for d,r in gold.items() if float(r)>0}
    raw=set(rankings['dense10'][:10])&positives; cov=set(rankings['coverage'][:10])&positives
    chosen=set(chosen); pool=set(pool)
    def categories(ds): return {'selected':len(ds&chosen),'inpool_unscored':len((ds&pool)-chosen),'outsidepool':len(ds-pool)}
    return {'metrics':scores,'gained':categories(cov-raw),'lost':categories(raw-cov),
            'top10_overlap_dense_coverage':len(set(rankings['dense10'][:10])&set(rankings['coverage'][:10]))}

def main(run_dir,qrels_path,base_path):
    run=Path(run_dir); base=Path(base_path)
    manifest=json.loads((run/'run_manifest.json').read_text())
    if manifest.get('qrels_read') is not False or len(manifest['qids'])!=200: raise ValueError('incomplete GPU run')
    frozen=Path(__file__).parent/'FROZEN_INPUTS.json'
    verify_digest(frozen,manifest['freeze_manifest_sha256'])
    frozen_data=json.loads(frozen.read_text())
    for name in ('evaluate_run.py','metrics.py','test_metrics.py','test_evaluate_run.py'):
        verify_digest(Path(__file__).parent/name,frozen_data['files'][name])
    expected_outputs={'selections.json','rankings.json','vectors.npy','SMOKE.json'}
    if manifest['physical_new_teacher_pairs']>0: expected_outputs.add('teacher_new.jsonl')
    if set(manifest['outputs_sha256'])!=expected_outputs: raise ValueError('incomplete output hash inventory')
    for name,expected in manifest['outputs_sha256'].items(): verify_digest(run/name,expected)
    if json.loads((run/'SMOKE.json').read_text()).get('status')!='pass': raise ValueError('GPU smoke did not pass')
    verify_digest(Path(__file__).parent/'query_ids.json',frozen_data['files']['query_ids.json'])
    qids=manifest['qids']
    if qids!=json.loads((Path(__file__).parent/'query_ids.json').read_text()): raise ValueError('frozen query ID order mismatch')
    rankings=json.loads((run/'rankings.json').read_text()); sel=json.loads((run/'selections.json').read_text())
    if set(qids)!=set(rankings) or set(qids)!=set(sel): raise ValueError('query alignment failure')
    verify_digest(base/'fiqa_doc_ids.json','0768cd2fdb2aa94e751d6a857cd82895607d3dc5c5382fb86133dc14e53f8674')
    verify_digest(base/'fiqa_top50.json','9182c53155fbb144186f5ad1d969f02966f0da01d43d08947c02775d88f8420c')
    corpus=set(json.loads((base/'fiqa_doc_ids.json').read_text()))
    for q in qids: validate_rankings(rankings[q],corpus)
    v=np.load(run/'vectors.npy')
    if v.shape!=(7,200,1024) or not np.isfinite(v).all(): raise ValueError('invalid vector array')
    tests=subprocess.run([sys.executable,'-m','unittest','test_metrics','test_evaluate_run'],cwd=Path(__file__).parent,capture_output=True,text=True)
    with (run/'evaluation_controls.json').open('x') as f:
        json.dump({'exit_code':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr},f,indent=2)
    if tests.returncode: raise ValueError('planted or corrupted-input controls failed')
    top=json.loads((base/'fiqa_top50.json').read_text()); pools={str(r['query_id']):[c['doc_id'] for c in r['candidates']] for r in top[:648]}
    verify_digest(qrels_path,QRELS_SHA)
    qrels=json.loads(Path(qrels_path).read_text())
    per={q:evaluate_one(q,rankings[q],qrels.get(q,{}),sel[q]['arms']['coverage'],pools[q]) for q in qids}
    def vec(a,m='ndcg10'):
        if a=='rankmatched_mean': return [np.mean([per[q]['metrics'][f'rankmatched_{i}'][m] for i in range(3)]) for q in qids]
        return [per[q]['metrics'][a][m] for q in qids]
    arms=ARMS+['rankmatched_mean']; summary={a:{m:float(np.mean(vec(a,m))) for m in ('ndcg10','recall10','recall100','mrr10')} for a in arms}
    contrasts={b:paired(vec('coverage'),vec(b)) for b in ('dense10','rankmatched_mean','bert_maxsim','rerank_dense10','rerank_coverage','dense')}
    promising=all(contrasts[b]['delta']>=.01 and contrasts[b]['ci95'][0]>0 for b in ('dense10','rankmatched_mean'))
    result={'queries':len(qids),'metrics':summary,'contrasts_coverage_minus':contrasts,
            'decision':'promising' if promising else 'not demonstrated',
            'gained':{k:sum(per[q]['gained'][k] for q in qids) for k in ('selected','inpool_unscored','outsidepool')},
            'lost':{k:sum(per[q]['lost'][k] for q in qids) for k in ('selected','inpool_unscored','outsidepool')},
            'mean_top10_overlap':float(np.mean([per[q]['top10_overlap_dense_coverage'] for q in qids])),
            'vector_norms_by_arm':np.linalg.norm(np.load(run/'vectors.npy'),axis=-1).mean(axis=-1).tolist()}
    for name,obj in [('per_query_metrics.json',per),('evaluation_summary.json',result)]:
        with (run/name).open('x') as f: json.dump(obj,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main(*sys.argv[1:])
