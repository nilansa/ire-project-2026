#!/usr/bin/env python3
"""Frozen-vector inference preparation. Gold labels enter only in score()."""
import argparse, csv, hashlib, json, math, os, platform, sys, time, uuid
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
import numpy as np
import alignment as a

HERE=Path(__file__).resolve().parent
SCRATCH=a.BASE/'ire-final-submission-20261115-1943/shared/baseline-work'
CAP=1800
SEED=20261115
BETA=(.05,.2,.5,1.)
ETA=(.05,.2,.8,3.2)
LAM=(.1,1.,10.)
TEACHERS=('medcpt','bge')
TEMPERATURES=(.5,4.)

def digest_obj(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,obj):
    temporary=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    with temporary.open('x') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    os.link(temporary,path)
    temporary.unlink()
def read(path): return json.loads(Path(path).read_text())
def code_hashes(): return {p.name:a.sha(p) for p in (HERE/'alignment.py',HERE/'harness.py',HERE/'test_harness.py',HERE/'PROTOCOL.md')}
def deadline(started_unix):
    if not 0 <= time.time()-started_unix <= CAP: raise TimeoutError('1800 second overall budget expired; run is incomplete')
def run_dir(name):
    p=Path(name).resolve()
    if not p.is_relative_to(SCRATCH.resolve()): raise ValueError('output must be under baseline-work')
    p.mkdir(parents=False,exist_ok=False)
    return p

def approved(review,alignment_path):
    r=read(review); al=read(alignment_path)
    if al['status']!='PASS': raise ValueError('alignment did not pass')
    if r.get('decision')!='APPROVED' or not r.get('independent_reviewer') or not r.get('reservation_id') or not r.get('independence_statement'):
        raise ValueError('independent approval, independence statement and reservation required')
    if len(r.get('qrels_sha256',''))!=64: raise ValueError('pre-metric qrels hash required')
    if r.get('code_hashes')!=code_hashes() or r.get('alignment_sha256')!=a.sha(Path(alignment_path)):
        raise ValueError('review does not cover current code, protocol and alignment')
    for name, expected in al['source_hashes_sha256'].items():
        if a.sha(Path(name))!=expected: raise ValueError(f'aligned source changed: {name}')
    return r

def start(args):
    approved(args.review,args.alignment)
    p=run_dir(args.out)
    save(p/'budget.json',{'started_unix':time.time(),'cap_seconds':CAP,'review_sha256':a.sha(Path(args.review)),'alignment_sha256':a.sha(Path(args.alignment)),'code_hashes':code_hashes()})

def budget(path,review,alignment_path):
    b=read(path)
    if b['review_sha256']!=a.sha(Path(review)) or b['alignment_sha256']!=a.sha(Path(alignment_path)) or b['code_hashes']!=code_hashes() or b['cap_seconds']!=CAP: raise ValueError('budget provenance mismatch')
    deadline(b['started_unix'])
    return b

def softmax(x):
    z=np.asarray(x,dtype=np.float64); z=z-z.max(); e=np.exp(z); return e/e.sum()

def teacher_diagnostics(teacher):
    diagnostics={}
    for temp in TEMPERATURES:
        probs=softmax(teacher[:20]/temp)
        diagnostics[f'teacher_entropy_t{temp:g}']=float(-np.sum(probs*np.log(np.maximum(probs,np.finfo(float).tiny))))
        diagnostics[f'teacher_max_probability_t{temp:g}']=float(probs.max())
    return diagnostics

def ranked(scores,keys,limit,trace=None,kind='sort'):
    scores=np.asarray(scores)
    if not np.isfinite(scores).all(): raise ValueError('non-finite score')
    limit=min(limit,len(scores))
    threshold=np.partition(scores,len(scores)-limit)[len(scores)-limit]
    candidates=np.flatnonzero(scores>=threshold)
    result=candidates[np.lexsort((keys[candidates],-scores[candidates]))[:limit]]
    if trace is not None: trace.append({'kind':kind,'input_scores':len(scores),'sorted_candidates':len(candidates),'requested':limit,'returned':len(result)})
    return result

def movement(q,docs,teacher,family,param,lam=None,stopping=False,temperature=.5):
    """Return adapted vector and feedback-gradient diagnostics, with no labels."""
    q=np.asarray(q,dtype=np.float64); docs=np.asarray(docs,dtype=np.float64)
    if family=='rocchio': return q+param*docs[:5].mean(axis=0),0.,0.,False,None,None
    probs=softmax(np.asarray(teacher[:20])/temperature)
    if family=='weighted': return q+param*(probs@docs[:20]),0.,0.,False,None,None
    if family!='tour': raise ValueError(family)
    student=softmax(docs[:20]@q)
    grad=(probs-student)@docs[:20]; norm=float(np.linalg.norm(grad)); direction=grad
    if lam is not None and norm>0:
        center=docs[:50]-docs[:50].mean(axis=0); meanvar=float(np.sum(center*center)/(len(center)*len(q)))
        if meanvar<=0: raise ValueError('zero local variance')
        # Economy SVD gives exact inverse in the local span plus shrinkage on its complement.
        _,s,vt=np.linalg.svd(center/np.sqrt(len(center)),full_matrices=False)
        shrink=lam*meanvar
        projection=vt@grad
        direction=grad/shrink + vt.T@(projection*(1/(s*s+shrink)-1/shrink))
        direction*=norm/np.linalg.norm(direction)
    cosine=float(np.clip(np.dot(grad,direction)/(norm*np.linalg.norm(direction)),-1,1)) if norm else 1.
    angle=float(np.degrees(np.arccos(cosine)))
    stopped=bool(stopping and float(teacher[0])==float(np.max(teacher[:20])))
    if stopped: return q,norm,angle,True,0.,0.
    target=param*(grad-.01*q)
    displacement=param*(direction-.01*q)
    if lam is not None:
        target_length=float(np.linalg.norm(target)); length=float(np.linalg.norm(displacement))
        if target_length==0: displacement=np.zeros_like(displacement)
        elif length==0: raise ValueError('cannot match nonzero TOUR displacement')
        else: displacement*=target_length/length
        if not np.isclose(np.linalg.norm(displacement),target_length,rtol=1e-12,atol=1e-15): raise ValueError('H2 displacement norm mismatch')
    reference=np.asarray(q+target,dtype=np.float32).astype(np.float64)-q
    size=float(np.linalg.norm(displacement)*np.linalg.norm(target))
    total_angle=float(np.degrees(np.arccos(np.clip(np.dot(displacement,target)/size,-1,1)))) if size else 0.
    return q+displacement,norm,angle,False,float(np.linalg.norm(reference)),total_angle

def arm_specs():
    arms=[{'name':'dense','family':'dense'}, {'name':'deep100','family':'deep100'}]
    for b in BETA: arms.append({'name':f'rocchio:{b:g}','family':'rocchio','param':b})
    for teacher in TEACHERS:
        for n in (20,50): arms.append({'name':f'rerank{n}:{teacher}','family':'rerank','teacher':teacher,'n':n})
        for temp in TEMPERATURES:
            for b in BETA: arms.append({'name':f'weighted:{teacher}:{b:g}:t{temp:g}','family':'weighted','teacher':teacher,'param':b,'temperature':temp})
            for e in ETA:
                for stop in (False,True): arms.append({'name':f'tour:{teacher}:{e:g}:stop{int(stop)}:t{temp:g}','family':'tour','teacher':teacher,'param':e,'stopping':stop,'temperature':temp})
                for l in LAM: arms.append({'name':f'h2:{teacher}:{e:g}:{l:g}:t{temp:g}','family':'tour','teacher':teacher,'param':e,'lam':l,'stopping':False,'temperature':temp})
    counterparts=[dict(s,name=s['name']+':pool',pool_only=True) for s in arms if s['family'] in ('rocchio','weighted','tour')]
    return arms+counterparts

def predict_one(q,corpus,initial,keys,teacher,spec,trace=None):
    family=spec['family']; first=list(map(int,initial)); f=corpus[initial]
    initial_ranking=ranked(corpus@q,keys,100 if family=='deep100' else 50,trace,'exact_search')
    if not np.array_equal(initial_ranking[:50],initial): raise ValueError('initial cache changed')
    if family=='dense': return first,first,0.,0.,False,0.,None,None
    if family=='deep100':
        return list(map(int,initial_ranking)),first,0.,0.,False,0.,None,None
    if family=='rerank':
        n=spec['n']; order=ranked(teacher[:n],keys[initial[:n]],n,trace,'teacher_sort')
        return list(map(int,initial[:n][order])),first,0.,0.,False,0.,None,None
    adapted,norm,angle,stopped,matched,total_angle=movement(q,f,teacher,family,spec['param'],spec.get('lam'),spec.get('stopping',False),spec.get('temperature',.5))
    if stopped: return first,first,norm,angle,True,0.,matched,total_angle
    adapted=np.asarray(adapted,dtype=np.float32)
    applied=float(np.linalg.norm(adapted.astype(np.float64)-q.astype(np.float64)))
    if spec.get('lam') is not None and abs(applied-matched)>8*np.finfo(np.float32).eps*max(float(np.linalg.norm(q)),matched): raise ValueError('float32 H2 step radius mismatch')
    if spec.get('pool_only',False):
        within=initial[ranked(f@adapted,keys[initial],50,trace,'pool_sort')]
        return list(map(int,within)),list(map(int,within)),norm,angle,False,applied,matched,total_angle
    scores=corpus@adapted
    full=ranked(scores,keys,50,trace,'exact_search')
    within=initial[ranked(scores[initial],keys[initial],50,trace,'pool_sort')]
    return list(map(int,full)),list(map(int,within)),norm,angle,stopped,applied,matched,total_angle

def inner_specs(selection):
    names=set(selection['selected_names'])|{'dense','deep100'}|{f'rerank{n}:{t}' for t in TEACHERS for n in (20,50)}
    names|={name+':pool' for name in selection['selected_names']}
    for name in selection['selected_names']:
        parts=name.split(':')
        if parts[0]=='h2': names.add(f'tour:{parts[1]}:{parts[2]}:stop0:{parts[4]}')
    result=[s for s in arm_specs() if s['name'] in names]
    if not names<={s['name'] for s in result}: raise ValueError('selection contains unknown arm')
    return result

def predict(args):
    started=time.monotonic(); review=approved(args.review,args.alignment); al=read(args.alignment)
    budget_start=budget(args.budget,args.review,args.alignment)['started_unix']
    specs=arm_specs(); selection=None
    if args.split=='inner':
        if not args.selection: raise ValueError('inner requires frozen development selection')
        selection=read(args.selection)
        if selection['review_sha256']!=a.sha(Path(args.review)) or selection['alignment_sha256']!=a.sha(Path(args.alignment)) or selection['budget_sha256']!=a.sha(Path(args.budget)) or selection['code_hashes']!=code_hashes(): raise ValueError('selection provenance mismatch')
        specs=inner_specs(selection)
    p=run_dir(args.out)
    emb=np.load(a.R8/'initial_embeddings.npz',allow_pickle=False); corpus=emb['corpus']; mine=emb['mine']
    qids=list(map(str,read(a.R8/'frozen_candidate_query_ids.json'))); qi={q:i for i,q in enumerate(qids)}
    ids=np.load(a.R8/'frozen_candidate_ids.npz',allow_pickle=False)['FaissFlatIP']; keys=np.load(a.R8/'document_tie_keys.npy',allow_pickle=False)
    tc=np.load(a.TEACHER/'teacher_scores_train505.npz',allow_pickle=False); ti={str(x):i for i,x in enumerate(tc['qid'])}
    teacher_arrays={name:tc[name] for name in TEACHERS}
    selected_qids=al['development_qids' if args.split=='dev' else 'inner_qids']
    order=np.random.default_rng(SEED).permutation(len(selected_qids)); selected_qids=[selected_qids[i] for i in order]
    predictions=p/'predictions.jsonl'; timings={s['name']:[] for s in specs}
    unique_pairs=set(); total_cache_lookups=0; total_cache_values=0
    with predictions.open('x') as out:
        for qid in selected_qids:
            deadline(budget_start); ix=qi[qid]; row=ids[ix]; q=mine[ix]
            for spec in specs:
                deadline(budget_start)
                family=spec['family']; logical=spec.get('n',20) if family=='rerank' else (20 if family in ('weighted','tour') else 0)
                teacher=teacher_arrays[spec['teacher']][ti[qid],:logical] if 'teacher' in spec else None
                if teacher is not None:
                    total_cache_lookups+=1; total_cache_values+=logical
                    unique_pairs.update((qid,spec['teacher'],int(doc)) for doc in row[:logical])
                # One untimed warmup, followed by three measured passes.
                vals=[]; events=[]; warm_events=[]
                previous=predict_one(q,corpus,row,keys,teacher,spec,warm_events)
                for _ in range(3):
                    t=time.perf_counter(); result=predict_one(q,corpus,row,keys,teacher,spec,events); vals.append(time.perf_counter()-t)
                    if result!=previous: raise ValueError('timing passes disagree')
                    previous=result
                timings[spec['name']].extend(vals)
                full,within,norm,angle,stopped,applied,matched,total_angle=result
                if len(set(full))!=len(full) or len(set(within))!=len(within): raise ValueError('duplicate prediction')
                scans=1 if family in ('dense','deep100','rerank') or stopped or spec.get('pool_only',False) else 2
                teacher_top1=int(row[int(np.argmax(teacher[:20]))]) if teacher is not None else None
                diagnostics=teacher_diagnostics(teacher) if teacher is not None else {}
                all_events=warm_events+events
                actual={'warmup_passes':1,'timed_passes':3,'exact_search_calls':sum(e['kind']=='exact_search' for e in all_events),'exact_search_returned_slots':sum(e['returned'] for e in all_events if e['kind']=='exact_search'),'sort_calls':len(all_events),'sorted_candidate_total':sum(e['sorted_candidates'] for e in all_events),'sort_input_total':sum(e['input_scores'] for e in all_events),'teacher_cache_row_lookups':int(teacher is not None),'teacher_cache_unique_values_in_view':logical,'warmup_events':warm_events,'timed_events':events}
                record={'qid':qid,'arm':spec['name'],'initial50':list(map(int,row)),'ranked':full,'within_initial':within,'gradient_norm':norm,'angle_degrees':angle,'complete_step_angle_degrees':total_angle,'applied_displacement_norm':applied,'matched_tour_displacement_norm':matched,'stopped':stopped,'teacher_top1_in_final20':int(teacher_top1 in full[:20]) if teacher_top1 is not None else None,'teacher_logical_pairs':logical,'teacher_fresh_calls':0,'full_corpus_scans':scans,'returned_slots':100 if family=='deep100' else (100 if scans==2 else 50),'unique_union':len(set(row)|set(full)),'physical_harness':actual,**diagnostics}
                out.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
    timing={k:{'n':len(v),'p50_seconds':float(np.percentile(v,50)),'p95_seconds':float(np.percentile(v,95))} for k,v in timings.items()}
    deadline(budget_start)
    manifest={'split':args.split,'prediction_sha256':a.sha(predictions),'review_sha256':a.sha(Path(args.review)),'alignment_sha256':a.sha(Path(args.alignment)),'budget_sha256':a.sha(Path(args.budget)),'selection_sha256':a.sha(Path(args.selection)) if args.selection else None,'code_hashes':code_hashes(),'specs':specs,'random_seed':SEED,'clock_seconds':time.monotonic()-started,'threads':1,'python':sys.version,'numpy':np.__version__,'machine':platform.platform(),'processor':platform.processor(),'timing_three_passes':timing,'timing_scope':'three timed warm CPU exact-search and update passes after one untimed warmup; excludes cache lookup, encoding and teacher inference','physical_teacher_cache_row_lookups_all_arms':total_cache_lookups,'physical_teacher_cache_values_in_views_all_arms':total_cache_values,'distinct_teacher_query_document_pairs_all_arms':len(unique_pairs),'data_sha256':al['source_hashes_sha256']}
    save(p/'manifest.json',manifest)

def metrics(ranked_docs,initial_docs,within_docs,qrels):
    missing=qrels is None
    qrels={} if missing else qrels
    rel={str(d):int(g) for d,g in qrels.items() if int(g)>0}; positives=set(rel)
    def dcg(seq,k): return sum((2**rel.get(str(x),0)-1)/math.log2(i+2) for i,x in enumerate(seq[:k]))
    ideal=sorted(rel.values(),reverse=True)[:10]
    denom=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    def rec(seq,k): return len(set(map(str,seq[:k]))&positives)/len(positives) if positives else None
    initial=set(map(str,initial_docs)); final=set(map(str,ranked_docs[:20])); final10=set(map(str,ranked_docs[:10])); all_returned=set(map(str,ranked_docs))
    oracle=sorted(initial,key=lambda x:(-rel.get(x,0),x))
    first=next((i+1 for i,x in enumerate(ranked_docs[:10]) if str(x) in positives),None)
    out={'ndcg10':dcg(ranked_docs,10)/denom if denom else None,'mrr10':(1/first if first else 0.) if positives else None,'recall10':rec(ranked_docs,10),'recall20':rec(ranked_docs,20),'initial_recall50':rec(initial_docs,50),'oracle_ndcg10_initial50':dcg(oracle,10)/denom if denom else None,'oracle_recall10_initial50':rec(oracle,10),'oracle_recall20_initial50':rec(oracle,20),'outside50_relevant_gained_at10':len((final10-initial)&positives),'outside50_relevant_gained_at20':len((final-initial)&positives),'relevant_lost_at10_vs_initial_top10':len((set(map(str,initial_docs[:10]))-final10)&positives),'relevant_lost_at20_vs_initial_top20':len((set(map(str,initial_docs[:20]))-final)&positives),'outside50_entries_at10':len(final10-initial),'outside50_entries_at20':len(final-initial),'union_recall':len((initial|all_returned)&positives)/len(positives) if positives else None,'judged_fraction10':sum(str(x) in qrels for x in ranked_docs[:10])/min(10,len(ranked_docs)),'within_ndcg10':dcg(within_docs,10)/denom if denom else None,'within_recall20':rec(within_docs,20),'initial_judged_count50':len(initial&set(qrels)),'initial_judged_fraction50':len(initial&set(qrels))/len(initial),'initial_judged_relevant_count50':len(initial&positives),'initial_pool_zero_positive':int(not (initial&positives)),'missing_qrels':int(missing),'all_qrels_zero_positive':int(not missing and not positives),'total_judged_relevant':len(positives)}
    feedback=set(map(str,initial_docs[:20])); regions={'f20':feedback,'u50_minus_f20':initial-feedback,'outside_u50':positives-initial}
    for region,docs in regions.items():
        region_positives=positives&docs
        out[f'judged_relevant_total_{region}']=len(region_positives)
        for k in (10,20):
            found=set(map(str,ranked_docs[:k]))&region_positives
            dense=set(map(str,initial_docs[:k]))&region_positives
            out[f'{region}_relevant_at{k}']=len(found)
            out[f'{region}_relevant_gained_at{k}_vs_dense']=len(found-dense)
            out[f'{region}_relevant_lost_at{k}_vs_dense']=len(dense-found)
    if missing:
        for key in out:
            if 'relevant' in key or key in ('judged_fraction10','initial_judged_count50','initial_judged_fraction50','initial_pool_zero_positive','all_qrels_zero_positive'):
                out[key]=None
    return out

def score(args):
    started=time.monotonic(); p=Path(args.predictions); m=read(p/'manifest.json')
    if m['prediction_sha256']!=a.sha(p/'predictions.jsonl') or m['code_hashes']!=code_hashes(): raise ValueError('prediction or source hash changed')
    if m['review_sha256']!=a.sha(Path(args.review)) or m['alignment_sha256']!=a.sha(Path(args.alignment)): raise ValueError('provenance mismatch')
    review=approved(args.review,args.alignment)
    budget_start=budget(args.budget,args.review,args.alignment)['started_unix']
    if m['budget_sha256']!=a.sha(Path(args.budget)): raise ValueError('budget mismatch')
    qrel_path=a.DATA/'qrels/train.tsv'; qrels={}
    if a.sha(qrel_path)!=review['qrels_sha256']: raise ValueError('qrels differ from preregistered hash')
    with qrel_path.open() as f:
        for line in csv.DictReader(f,delimiter='\t'):
            qid=line['query-id']; doc=line['corpus-id']; grade=int(line['score']); qrels.setdefault(qid,{})[doc]=grade
    docs=[str(x['_id']) for x in a.readlines(a.DATA/'corpus.jsonl')]
    outdir=run_dir(args.out); rows=[]
    with (p/'predictions.jsonl').open() as source:
        for line in source:
            deadline(budget_start); x=json.loads(line); q=x['qid']; mapped=lambda seq:[docs[i] for i in seq]
            vals=metrics(mapped(x['ranked']),mapped(x['initial50']),mapped(x['within_initial']),qrels.get(q))
            physical=x['physical_harness']
            fields=('gradient_norm','angle_degrees','complete_step_angle_degrees','applied_displacement_norm','matched_tour_displacement_norm','stopped','teacher_top1_in_final20','teacher_logical_pairs','teacher_fresh_calls','full_corpus_scans','returned_slots','unique_union')
            diag={f'teacher_{metric}_t{temp:g}':x.get(f'teacher_{metric}_t{temp:g}') for temp in TEMPERATURES for metric in ('entropy','max_probability')}
            rows.append({'qid':q,'arm':x['arm'],**vals,**{k:x[k] for k in fields},**diag,**{f'physical_{k}':physical[k] for k in ('warmup_passes','timed_passes','exact_search_calls','exact_search_returned_slots','sort_calls','sorted_candidate_total','sort_input_total','teacher_cache_row_lookups','teacher_cache_unique_values_in_view')}})
    if not rows: raise ValueError('empty predictions')
    with (outdir/'per_query.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    byarm={}
    for row in rows: byarm.setdefault(row['arm'],{})[row['qid']]=row
    baseline=byarm['dense']; rng=np.random.default_rng(SEED); summary={}
    all_qids=sorted(baseline); qids=[q for q in all_qids if baseline[q]['ndcg10'] is not None]
    invalid=[q for q in all_qids if baseline[q]['ndcg10'] is None]
    idx=rng.integers(0,len(qids),size=(2000,len(qids))) if qids else None
    for arm,items in byarm.items():
        if set(items)!=set(all_qids): raise ValueError('arms have different qids')
        differences=np.array([items[q]['ndcg10']-baseline[q]['ndcg10'] for q in qids]); draws=differences[idx].mean(axis=1) if qids else None
        aggregate={}; totals={}
        for k in items[all_qids[0]]:
            if k in ('qid','arm'): continue
            values=[items[q][k] for q in all_qids if items[q][k] is not None]
            aggregate[k]=float(np.mean(values)) if values else None
            if k in ('total_judged_relevant','teacher_logical_pairs','teacher_fresh_calls','full_corpus_scans','returned_slots','unique_union') or any(s in k for s in ('relevant_', 'judged_count','zero_positive','missing_qrels','physical_')): totals[k]=float(np.sum(values)) if values else None
        summary[arm]={'means':aggregate,'totals':totals,'mean_ndcg10':aggregate['ndcg10'],'paired_dense_delta_ndcg10':float(differences.mean()) if qids else None,'paired_qid_bootstrap_95':list(map(float,np.percentile(draws,[2.5,97.5]))) if qids else None,'unscorable_qids':invalid}
        parts=arm.split(':')
        if parts[0]=='h2' and not arm.endswith(':pool'):
            comparator=f'tour:{parts[1]}:{parts[2]}:stop0:{parts[4]}'
            if comparator in byarm and qids:
                diff=np.array([items[q]['ndcg10']-byarm[comparator][q]['ndcg10'] for q in qids])
                summary[arm]['matched_tour']={'arm':comparator,'delta_ndcg10':float(diff.mean()),'paired_qid_bootstrap_95':list(map(float,np.percentile(diff[idx].mean(axis=1),[2.5,97.5])))}
    save(outdir/'summary.json',summary)
    deadline(budget_start)
    save(outdir/'manifest.json',{'split':m['split'],'prediction_manifest_sha256':a.sha(p/'manifest.json'),'prediction_sha256':m['prediction_sha256'],'per_query_sha256':a.sha(outdir/'per_query.csv'),'summary_sha256':a.sha(outdir/'summary.json'),'qrels_sha256':a.sha(qrel_path),'review_sha256':m['review_sha256'],'alignment_sha256':m['alignment_sha256'],'budget_sha256':m['budget_sha256'],'code_hashes':code_hashes(),'bootstrap_draws':2000,'seed':SEED,'status':'exploratory'})

def choose_winners(summary,rows):
    groups={}
    for name in summary:
        parts=name.split(':'); family=parts[0]
        if family not in ('rocchio','weighted','tour','h2') or name.endswith(':pool'): continue
        key=family if family=='rocchio' else family+':'+parts[1]
        groups.setdefault(key,[]).append(name)
    expected={'rocchio'}|{f'{family}:{teacher}' for family in ('weighted','tour','h2') for teacher in TEACHERS}
    if set(groups)!=expected: raise ValueError('missing development family')
    if any(summary[name].get('unscorable_qids') for name in summary): raise ValueError('unscorable qrels prohibit selection')
    chosen=[]; families={}; dense=summary['dense']['mean_ndcg10']
    # Ties prefer fewer scans, then the configured step, then the stable arm name.
    for group,names in sorted(groups.items()):
        def key(n):
            subset=[r for r in rows if r['arm']==n]; cost=np.mean([float(r['full_corpus_scans']) for r in subset]); parts=n.split(':')
            step=float(parts[1] if parts[0]=='rocchio' else parts[2])
            return (-summary[n]['mean_ndcg10'],cost,step,n)
        best=min(names,key=key)
        chosen.append(best)
        families[group]={'dev_winner':best,'dev_ndcg10':summary[best]['mean_ndcg10'],'deployment_choice':best if summary[best]['mean_ndcg10']>dense else 'dense'}
    return chosen,families

def select(args):
    p=Path(args.scored); m=read(p/'manifest.json')
    b=read(args.budget); deadline(b['started_unix'])
    if m['budget_sha256']!=a.sha(Path(args.budget)) or b['code_hashes']!=code_hashes(): raise ValueError('budget mismatch')
    if m['split']!='dev' or m['code_hashes']!=code_hashes() or m['per_query_sha256']!=a.sha(p/'per_query.csv') or m['summary_sha256']!=a.sha(p/'summary.json'): raise ValueError('invalid development score provenance')
    summary=read(p/'summary.json'); rows=list(csv.DictReader((p/'per_query.csv').open()))
    chosen,families=choose_winners(summary,rows)
    p_out=run_dir(args.out)
    deadline(b['started_unix'])
    save(p_out/'selection.json',{'selected_names':chosen,'families':families,'fallback':'deployment choice is dense on equal or lower development nDCG10; every dev winner still runs on INNER','criterion':'max dev mean nDCG10; ties fewer scans, smaller configured step, lexical name','development_per_query_sha256':m['per_query_sha256'],'development_summary_sha256':m['summary_sha256'],'review_sha256':m['review_sha256'],'alignment_sha256':m['alignment_sha256'],'budget_sha256':m['budget_sha256'],'code_hashes':code_hashes(),'status':'exploratory'})

def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='cmd',required=True)
    for cmd in ('start','predict','score'):
        s=sub.add_parser(cmd); s.add_argument('--review',required=True); s.add_argument('--alignment',required=True); s.add_argument('--out',required=True)
        if cmd=='predict': s.add_argument('--split',choices=('dev','inner'),required=True); s.add_argument('--selection')
        if cmd=='score': s.add_argument('--predictions',required=True)
        if cmd!='start': s.add_argument('--budget',required=True)
    s=sub.add_parser('select'); s.add_argument('--scored',required=True); s.add_argument('--out',required=True); s.add_argument('--budget',required=True)
    args=parser.parse_args(); globals()[args.cmd](args)
if __name__=='__main__': main()
