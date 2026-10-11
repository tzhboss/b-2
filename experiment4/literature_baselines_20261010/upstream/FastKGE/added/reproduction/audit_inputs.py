"""Independent raw-input and released precomputed-ID audit; never rewrites data."""
import hashlib,json,pathlib,collections
base=pathlib.Path(__file__).resolve().parent
c=json.loads((base/'config.json').read_text());repo=(base/c['repo_root']).resolve();data=repo/'data'/c['dataset']
entities={};relations={};rows=[];files=[];seen_triples=set()
for i in range(5):
    folder=data/str(i);splits={}
    for split in ['train','valid','test']:
        path=folder/(split+'.txt');triples=[tuple(line.split()) for line in path.read_text().splitlines()]
        assert all(len(t)==3 for t in triples);splits[split]=triples
        for h,r,t in triples:
            for e in [h,t]:
                if e not in entities:entities[e]=len(entities)
            if r not in relations:
                k=len(relations);relations[r]=k;relations[r+'_inv']=k+1
    ent_file=dict(line.split() for line in (folder/'entity2id.txt').read_text().splitlines());rel_file=dict(line.split() for line in (folder/'relation2id.txt').read_text().splitlines())
    assert {k:int(v) for k,v in ent_file.items()}==entities
    assert {k:int(v) for k,v in rel_file.items()}==relations
    tid=[tuple(map(int,line.split())) for line in (folder/'train_id.txt').read_text().splitlines()]
    assert tid==[(entities[h],relations[r],entities[t]) for h,r,t in splits['train']]
    f='train_distance_nodes2.txt' if c['method']=='STARK' else 'train_distance_nodes.txt'
    feature=[line.split() for line in (folder/f).read_text().splitlines()]
    assert all(len(x)==3 and 0<=int(x[0])<len(entities) for x in feature)
    counts=collections.Counter(int(x[0]) for x in feature);assert all(v==1 for v in counts.values())
    feature_ids=set(counts);current=set().union(*map(set,splits.values()))
    overlap={a+'_'+b:len(set(splits[a])&set(splits[b])) for a,b in [('train','valid'),('train','test'),('valid','test')]}
    rows.append({'snapshot':i,'entities':len(entities),'relations_including_inverse':len(relations),'split_rows':{k:len(v) for k,v in splits.items()},'within_snapshot_split_overlap':overlap,'triples_seen_in_previous_snapshots':len(current&seen_triples),'feature_rows':len(feature),'feature_columns':3,'feature_ids_missing':len(entities)-len(feature_ids),'model_feature_column':1 if c['method']=='STARK' else [1,2]})
    seen_triples |= current
    for file in ['train.txt','valid.txt','test.txt','entity2id.txt','relation2id.txt','train_id.txt',f]:
        path=folder/file;files.append({'path':str(path.relative_to(repo)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
aggregate=hashlib.sha256(''.join(x['path']+' '+x['sha256']+'\n' for x in files).encode()).hexdigest()
raw=hashlib.sha256(''.join(x['path']+' '+x['sha256']+'\n' for x in files if pathlib.Path(x['path']).name in ['train.txt','valid.txt','test.txt']).encode()).hexdigest()
out={'kind':'INPUT_AUDIT_NOT_TRAINING_RESULT','dataset':'FB_CKGE','mapping_exact_match_all_five_snapshots':True,'input_manifest_sha256':aggregate,'raw_splits_manifest_sha256':raw,'snapshots':rows,'files':files}
(base/'input_manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2))
