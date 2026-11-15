import math
import unittest
import numpy as np
import harness as h

class RequiredChecks(unittest.TestCase):
    def test_ideal_uses_all_relevant_documents(self):
        result=h.metrics(['a','x'],['a','x'],['a','x'],{'a':1,'outside':1})
        self.assertAlmostEqual(result['ndcg10'],1/(1+1/math.log2(3)))
        self.assertEqual(result['initial_recall50'],.5)
        self.assertLess(result['oracle_ndcg10_initial50'],1)

    def test_promotion_and_discovery_are_distinct(self):
        corpus=np.array([[1-i/100,1] if i<5 else [1-i/100,0] for i in range(51)]+[[.1,20]],dtype=np.float32)
        q=np.array([1,0],dtype=np.float32); keys=np.arange(len(corpus)); initial=h.ranked(corpus@q,keys,50)
        full,within,*_=h.predict_one(q,corpus,initial,keys,None,{'family':'rocchio','param':.5})
        pool,*_=h.predict_one(q,corpus,initial,keys,None,{'family':'rocchio','param':.5,'pool_only':True})
        self.assertEqual(pool,within)
        self.assertNotIn(51,within)
        self.assertIn(51,full[:20])
        m=h.metrics(list(map(str,full)),list(map(str,initial)),list(map(str,within)),{'51':1})
        self.assertEqual(m['outside50_relevant_gained_at20'],1)
        self.assertGreater(m['outside50_entries_at20'],0)
        self.assertEqual(m['within_recall20'],0)
        self.assertEqual(m['union_recall'],1)

    def test_labels_cannot_change_prediction_and_ties_are_stable(self):
        corpus=np.array([[1.,0.]]*55,dtype=np.float32); q=np.array([1.,0.],dtype=np.float32); keys=np.arange(55)[::-1]
        initial=h.ranked(corpus@q,keys,50)
        spec={'family':'rocchio','param':.2}
        labels1={'54':1}; labels2={'0':1}
        a=h.predict_one(q,corpus,initial,keys,None,spec)
        first=h.metrics(list(map(str,a[0])),list(map(str,initial)),list(map(str,a[1])),labels1)
        b=h.predict_one(q,corpus,initial,keys,None,spec)
        second=h.metrics(list(map(str,b[0])),list(map(str,initial)),list(map(str,b[1])),labels2)
        self.assertEqual(a,b)
        self.assertEqual(a[0],list(range(54,4,-1)))
        self.assertEqual(len(set(a[0])),50)
        self.assertNotEqual(first['ndcg10'],second['ndcg10'])

    def test_h2_matches_complete_displacement_with_decay(self):
        rng=np.random.default_rng(713)
        corpus=rng.normal(size=(60,3)).astype(np.float32)
        q=np.array([.7,-.2,.5],dtype=np.float32); keys=np.arange(60)
        initial=h.ranked(corpus@q,keys,50); teacher=np.linspace(-3,2,20)
        base={'family':'tour','param':.8,'temperature':4.}
        t=h.predict_one(q,corpus,initial,keys,teacher,base)
        v=h.predict_one(q,corpus,initial,keys,teacher,{**base,'lam':1.})
        self.assertGreater(v[3],1.)
        self.assertGreater(v[7],1.)
        self.assertAlmostEqual(v[5],t[5],delta=5e-7)
        self.assertAlmostEqual(v[6],t[5],delta=1e-12)

    def test_losing_dev_winners_still_selected(self):
        specs=[s for s in h.arm_specs() if not s['name'].endswith(':pool')]
        summary={s['name']:{'mean_ndcg10':.1,'unscorable_qids':[]} for s in specs}
        summary['dense']['mean_ndcg10']=.9
        rows=[{'arm':s['name'],'full_corpus_scans':2} for s in specs]
        selected,families=h.choose_winners(summary,rows)
        self.assertEqual(len(selected),7)
        self.assertEqual(len(families),7)
        self.assertTrue(all(f['deployment_choice']=='dense' for f in families.values()))
        self.assertEqual(set(selected),{f['dev_winner'] for f in families.values()})
        inner={s['name'] for s in h.inner_specs({'selected_names':selected})}
        self.assertTrue(set(selected)<=inner)
        self.assertTrue({n+':pool' for n in selected}<=inner)

    def test_three_regions_and_missing_judgments(self):
        initial=list(map(str,range(50)))
        ranked=['55','21','15']+[str(i) for i in range(2,50) if i not in (15,21)]
        qrels={'0':1,'15':1,'21':1,'55':1,'30':0}
        m=h.metrics(ranked,initial,initial,qrels)
        self.assertEqual([m[f'judged_relevant_total_{x}'] for x in ('f20','u50_minus_f20','outside_u50')],[2,1,1])
        self.assertEqual([m[f'{x}_relevant_gained_at10_vs_dense'] for x in ('f20','u50_minus_f20','outside_u50')],[1,1,1])
        self.assertEqual(m['f20_relevant_lost_at10_vs_dense'],1)
        self.assertEqual(m['initial_judged_count50'],4)
        missing=h.metrics(ranked,initial,initial,None)
        self.assertEqual(missing['missing_qrels'],1)
        self.assertIsNone(missing['initial_pool_zero_positive'])
        self.assertIsNone(missing['ndcg10'])
        zero=h.metrics(ranked,initial,initial,{'30':0})
        self.assertEqual(zero['all_qrels_zero_positive'],1)
        self.assertEqual(zero['initial_pool_zero_positive'],1)
        self.assertIsNone(zero['ndcg10'])

    def test_physical_search_limit_and_temperature_grid(self):
        corpus=np.arange(240,dtype=np.float32).reshape(120,2); q=np.array([1.,0.],dtype=np.float32); keys=np.arange(120)
        initial=h.ranked(corpus@q,keys,50); dense_trace=[]; deep_trace=[]
        h.predict_one(q,corpus,initial,keys,None,{'family':'dense'},dense_trace)
        h.predict_one(q,corpus,initial,keys,None,{'family':'deep100'},deep_trace)
        self.assertEqual((dense_trace[0]['requested'],dense_trace[0]['returned']),(50,50))
        self.assertEqual((deep_trace[0]['requested'],deep_trace[0]['returned']),(100,100))
        self.assertEqual((dense_trace[0]['sorted_candidates'],deep_trace[0]['sorted_candidates']),(50,100))
        logits=np.arange(20,dtype=float); diag=h.teacher_diagnostics(logits)
        self.assertGreater(diag['teacher_entropy_t4'],diag['teacher_entropy_t0.5'])
        self.assertLess(diag['teacher_max_probability_t4'],diag['teacher_max_probability_t0.5'])
        specs=h.arm_specs()
        self.assertEqual(len(specs),206)
        self.assertEqual({s['temperature'] for s in specs if 'temperature' in s},{.5,4.})

if __name__=='__main__': unittest.main()
