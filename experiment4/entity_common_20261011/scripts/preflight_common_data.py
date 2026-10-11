"""Read-only, audit-bound data feasibility for common novel entity tasks."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
    cfg=json.loads((root/'configs/data-preflight.json').read_text());base=(root/cfg['source_root']).resolve();data=base/cfg['dataset'];ap=base/cfg['source_audit'];audit=json.loads(ap.read_text());assert not audit['failures'] and not audit['missing']
    bound={str(ap):sha(ap)};tables={};stats=[]
    for i,expect in zip(cfg['snapshots'],cfg['expected_entities']):
        def read(name):
            p=data/str(i)/name;digest=sha(p);assert digest==audit['data']['sha256'][f'{i}/{name}'];bound[str(p)]=digest;return p.read_text().splitlines()
        E={};R={}
        for name,mapping in [('entity2id.txt',E),('relation2id.txt',R)]:
            for line in read(name):key,val=line.split();mapping[key]=int(val)
        assert len(E)==expect and sorted(E.values())==list(range(expect))
        splits={}
        for split in ['train','valid','test']:
            triples=[]
            for line in read(split+'.txt'):
                h,r,t=line.split();triples.append((E[h],R[r],E[t]))
            assert len(set(triples))==len(triples);splits[split]=triples
        tables[i]=(E,R,splits)
        stats.append(dict(snapshot=i,entities=len(E),relations=len(R),counts={k:len(v) for k,v in splits.items()},within_snapshot_overlap={a+'_'+b:len(set(splits[a])&set(splits[b])) for a,b in [('train','valid'),('train','test'),('valid','test')]}))
    e0,r0,s0=tables[0];e1,r1,s1=tables[1];assert all(e1[k]==v for k,v in e0.items()) and r0==r1
    cut=len(e0);novel=set(range(cut,len(e1)));new={k:[f for f in v if f[0]>=cut or f[2]>=cut] for k,v in s1.items()}
    touched={v for h,r,t in new['train'] for v in [h,t] if v>=cut}
    eval_coverage={}
    for split in ['valid','test']:
        no=[f for f in new[split] if any(v in novel-touched for v in [f[0],f[2]])]
        eval_coverage[split]=dict(novel_fact_count=len(new[split]),queries_head_tail=2*len(new[split]),facts_touching_no_train_support=len(no))
    allold=set(s0['train']+s0['valid']+s0['test'])
    overlaps={k:len(set(v)&allold) for k,v in new.items()};assert all(v==0 for v in overlaps.values())
    for p,d in bound.items():assert sha(Path(p))==d
    result=dict(status='PASS_DATA_FEASIBILITY_ONLY',snapshots=stats,new_entities=len(novel),new_entities_with_increment_train_support=len(touched),new_entities_without_train_support=len(novel-touched),
        increment_counts={k:len(v) for k,v in new.items()},new_eval_coverage=eval_coverage,new_vs_old_fact_overlaps=overlaps,
        before_old_candidates=cut,after_all_method_candidates=len(e1),bound_input_sha256=bound,script_sha256=sha(Path(__file__)),
        model_training_started=False,limitations=['Existing development source data previously exposed; no heldout or training claim.', 'Unsupported new entities must remain candidates; report coverage strata, do not remove by outcome.'])
    out=root/'artifacts/data-preflight';out.mkdir(parents=True,exist_ok=False);(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
