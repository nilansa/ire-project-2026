#!/usr/bin/env python3
"""Render the fixed descriptive figures from the packaged summaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).parent; s=json.loads((D/'results/20261115-baseline/inner-score/summary.json').read_text())
arms=['dense','rocchio:0.05','rerank20:medcpt','weighted:medcpt:1:t4','tour:medcpt:0.8:stop0:t4','h2:medcpt:0.8:10:t4','rerank50:medcpt']
labels=['Dense (0 pairs)','PRF (0 pairs)','Rerank20 (20)','Weighted (20)','TOUR (20)','H2 (20)','Rerank50 (50; diagnostic)']
fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True,layout='constrained')
for ax,metric,title in zip(axes,['ndcg10','recall20'],['nDCG@10','Recall@20']):
 vals=[s[a]['means'][metric] for a in arms]; bars=ax.barh(labels,vals,color=['#66869a']*6+['#d17b42']);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(title);ax.set_xlabel('Mean score');ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('INNER: DEV-selected settings; descriptive, previously exposed (n=100)',fontsize=13)
fig.text(.5,-.02,'Logical teacher pairs shown; rerank50 uses higher feedback. Latency is cache-only, not end-to-end.',ha='center',fontsize=9)
(D/'plots').mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(D/f'plots/inner-methods.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained');x=[0,1];ax.bar(x,[504,526],label='Covered judged-relevant pairs',color='#4d8197');ax.bar(x,[64,42],bottom=[504,526],label='Missing judged-relevant pairs',color='#dba575')
for i,(a,b) in enumerate([(504,64),(526,42)]):ax.text(i,a/2,str(a),ha='center',color='white',fontsize=12);ax.text(i,a+b/2,str(b),ha='center',fontsize=11)
ax.set_xticks(x,['Initial 50','Initial 100']);ax.set_ylabel('Judged-relevant query–document pairs');ax.set_ylim(0,700);ax.set_title('Pool coverage across 505 previously exposed queries');ax.legend(loc='upper center', ncol=2, fontsize=9);ax.text(.02,-.16,'568 judged-relevant pairs total; unjudged documents are not known negatives.',transform=ax.transAxes,fontsize=9)
for ext in ['png','svg']:fig.savefig(D/f'plots/pool-coverage.{ext}',dpi=180,bbox_inches='tight')
# Matplotlib emits trailing spaces in SVG path data; remove them without changing geometry.
for p in (D/'plots').glob('*.svg'):
 p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
