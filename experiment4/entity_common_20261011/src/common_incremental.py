"""Controlled novel-entity training, native-method bridges and raw rank export.

Native source is imported in one isolated process per method; fitting snapshots
contain TRAIN only. TEST has a separate reader, used only after checkpoint selection.
"""
import ast, importlib, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace as NS
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common_training import EmbeddingModel, sha


class Data:
    def __init__(self, root, cfg):
        self.root = (root / cfg['source_root']).resolve()
        self.folder = self.root / cfg['dataset']
        self.audit = json.loads((self.root / cfg['source_audit']).read_text())
        assert not self.audit['failures'] and not self.audit['missing']
        self.bound = {}
        self.entities = {}; self.relations = {}
        for s in (0, 1):
            self.entities[s] = dict((a, int(b)) for a, b in (x.split() for x in self.read(s, 'entity2id.txt')))
            self.relations[s] = dict((a, int(b)) for a, b in (x.split() for x in self.read(s, 'relation2id.txt')))
        assert len(self.entities[0]) == 7505 and len(self.entities[1]) == 11258
        assert self.relations[0] == self.relations[1]
        assert all(self.entities[1][k] == v for k, v in self.entities[0].items())

    def read(self, snapshot, name):
        p = self.folder / str(snapshot) / name; digest = sha(p)
        assert digest == self.audit['data']['sha256'][f'{snapshot}/{name}'], p
        self.bound[str(p)] = digest
        return p.read_text().splitlines()

    def triples(self, s, split):
        rows = []
        for line in self.read(s, split + '.txt'):
            h, r, t = line.split()
            h, r, t = self.entities[s][h], self.relations[s][r], self.entities[s][t]
            assert r % 2 == 0
            rows.extend([(h, r, t), (t, r+1, h)])
        return torch.tensor(rows, dtype=torch.long)


def defaults(path):
    result = {}
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument':
            kw = {k.arg: k.value for k in n.keywords}
            key = ast.literal_eval(kw['dest']) if 'dest' in kw else ast.literal_eval(n.args[0]).lstrip('-')
            if 'default' in kw: result[key] = ast.literal_eval(kw['default'])
    return result


def make_train_features(train0, train1, method, folder):
    # Exact in-place scan distance used by author nodes_sort.py, not BFS.
    seen = set(train0[:, 0].tolist()) | set(train0[:, 2].tolist())
    values = {v: (0, 0.) for v in seen}; distance = 0
    facts = train1[::2].tolist()
    while True:
        previous = len(seen)
        for h, _, t in facts:
            if h in seen and t not in seen: values[t] = (distance+1, 0.); seen.add(t)
            elif h not in seen and t in seen: values[h] = (distance+1, 0.); seen.add(h)
        if len(seen) == previous: break
        distance += 1
    neighbours = {}
    for h, _, t in facts:
        neighbours.setdefault(h, set()).add(t); neighbours.setdefault(t, set()).add(h)
    for v, adjacent in neighbours.items():
        # Self loops count twice in NetworkX Graph.degree; preserve author definition.
        degree = (len(adjacent) + (v in adjacent)) / (len(neighbours)-1)
        values[v] = (values.get(v, (100, 0))[0], degree)
    folder.mkdir(parents=True, exist_ok=True)
    name = 'train_distance_nodes.txt' if method == 'FastKGE' else 'train_distance_nodes2.txt'
    if method == 'FastKGE':
        text = ''.join(f'{v}\t{d}\t{s}\n' for v, (d, s) in values.items())
    else:
        # STARK reader consumes field[1], i.e. distance, even with three-column source.
        # Only the consumed fields are emitted; no unused PageRank computation.
        text = ''.join(f'{v}\t{d}\n' for v, (d, _) in values.items())
    (folder / name).write_text(text)
    return folder / name


def setup(root, cfg, job, output):
    seed = job['seed']; torch.set_num_threads(cfg['cpu_threads_per_job'])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); assert torch.cuda.is_available()
    data = Data(root, cfg); train0 = data.triples(0, 'train'); train1 = data.triples(1, 'train')
    assert ((train1[:, 0] >= 7505) | (train1[:, 2] >= 7505)).all()
    bpath = root / 'artifacts/base' / f"{job['base']}-s{seed}" / 'best.pt'
    baudit = json.loads(bpath.with_name('audit.json').read_text())
    assert not baudit.get('failures'), baudit
    base = torch.load(bpath, map_location=device, weights_only=True)
    E0 = base['state_dict']['entity.weight']; R0 = base['state_dict']['relation.weight']
    gen = torch.Generator(device=device).manual_seed(seed + cfg['expansion_seed_offset'])
    dim = E0.shape[1]; dtype = E0.dtype
    E1 = torch.empty((11258, dim), device=device, dtype=dtype)
    if job['kind'] == 'TransE': nn.init.xavier_normal_(E1, generator=gen)
    else: nn.init.normal_(E1, std=.12, generator=gen)
    E1[:7505] = E0
    args = None; native_bound = {}; method = job['method']
    if method in ('LKGE', 'FastKGE', 'STARK'):
        source = (data.root / cfg['native_sources'][method]).resolve()
        for f in sorted((source / 'src').rglob('*.py')): native_bound[str(f)] = sha(f)
        args = NS(**defaults(source / 'src/parse_args.py'))
        args.__dict__.update(job.get('native_options', {}))
        args.__dict__.update(snapshot=0, snapshot_valid=1, snapshot_test=1, epoch=0,
                             device=device, emb_dim=dim, neg_ratio=cfg['negatives'], margin=cfg['margin'], dataset='COMMON')
        sys.path.insert(0, str(source))
        for dep in cfg['dependency_paths']: sys.path.insert(0, str((data.root / dep).resolve()))
        snapshots = {}
        for s, train in ((0, train0), (1, train1)):
            snapshots[s] = NS(num_ent=(7505 if s == 0 else 11258), num_rel=474,
                              train=train[::2].tolist(),
                              train_all=(train0[::2] if s == 0 else torch.cat([train0[::2], train1[::2]])).tolist(),
                              edge_index=train[:, [0, 2]].T.to(device),
                              edge_type=train[:, 1].to(device))
        train_hr2t = defaultdict(set)
        # Original STARK loader adds only original-direction TRAIN facts.
        for h, r, t in torch.cat([train0[::2], train1[::2]]).tolist(): train_hr2t[h, r].add(t)
        kg = NS(snapshots=snapshots, train_hr2t=train_hr2t)
        if method in ('FastKGE', 'STARK'):
            make_train_features(train0, train1, method, output / 'native-cwd/data/COMMON/1')
            os.chdir(output / 'native-cwd')
        module = 'LKGE' if method == 'LKGE' else 'LoraKGE_Layers'
        model = importlib.import_module('src.model.' + module).TransE(args, kg)
        with torch.no_grad(): model.ent_embeddings.weight.copy_(E0); model.rel_embeddings.weight.copy_(R0)
        model.switch_snapshot(); args.snapshot = 1
        with torch.no_grad():
            if method == 'LKGE': model.ent_embeddings.weight[7505:].copy_(E1[7505:])
            else:
                # Author gather is [old IDs]+ordered IDs. Inverse placement gives
                # each canonical entity exactly the common initialization.
                order = torch.tensor(model.new_ordered_entities, device=device) - 7505
                physical = torch.empty_like(E1[7505:]); physical[order] = E1[7505:]
                begin = 0
                for layer in model.lora_ent_embeddings_list:
                    count = min(len(layer.weight), len(physical)-begin)
                    if count > 0: layer.weight[:count].copy_(physical[begin:begin+count])
                    begin += len(layer.weight)
        extract = lambda: extract_native(model, method)
        predict = lambda q: model.predict(q[:, 0], q[:, 1], stage='Valid')
    else:
        model = EmbeddingModel(11258, 474, dim, job['kind'], dtype).to(device)
        with torch.no_grad(): model.entity.weight.copy_(E1); model.relation.weight.copy_(R0)
        if method == 'NewEntityOnly':
            model.entity.weight.register_hook(lambda grad: torch.cat([torch.zeros_like(grad[:7505]), grad[7505:]]))
            model.relation.weight.requires_grad_(False)
        extract = lambda: (model.entity.weight, model.relation.weight)
        predict = model.candidate_scores
    E, R = extract()
    assert E.shape == E1.shape and R.shape == R0.shape
    assert torch.equal(E, E1) and torch.equal(R, R0), 'Common initialized effective embedding mismatch'
    q = train1[:16].to(device)
    model.eval()
    with torch.no_grad():
        native = predict(q); common = scores(E, R, q, job['kind'])
        assert torch.allclose(native, common, atol=1e-12, rtol=1e-12)
    bound = dict(data.bound, **native_bound); bound[str(bpath)] = sha(bpath)
    return data, model, args, extract, predict, train0, train1, E0, R0, E1.detach().clone(), bound


def extract_native(model, method):
    if method == 'LKGE': return model.embedding('Train')
    E, R = model.embedding('Train'); newE, newR = model.get_lora_embeddings()
    ix = list(range(model.kg.snapshots[0].num_ent)) + model.new_ordered_entities
    return torch.cat([E, newE])[ix], torch.cat([R, newR])


def scores(E, R, q, kind):
    if kind == 'TransE':
        e = F.normalize(E, dim=-1); r = F.normalize(R, dim=-1)
        return torch.sigmoid(9 - (e[q[:, 0], None] + r[q[:, 1], None] - e[None]).abs().sum(-1))
    return (E[q[:, 0]] * R[q[:, 1]]) @ E.T


@torch.no_grad()
def ranks(E, R, queries, kind, batch=16):
    values = []
    for part in queries.split(batch):
        q = part.to(E.device); s = scores(E, R, q, kind)
        gold = s[torch.arange(len(q), device=E.device), q[:, 2]]
        values.append((1 + (s > gold[:, None]).sum(1)).cpu().numpy())
    return np.concatenate(values)


def metrics(rank):
    return dict(queries=len(rank), raw_mrr=float((1/rank).mean()),
                hits1=float((rank <= 1).mean()), hits10=float((rank <= 10).mean()))


def train(root, cfg, job, preflight=False):
    stage = cfg.get('preflight_artifact_directory', 'preflight') if preflight else 'incremental'
    output = root / 'artifacts' / stage / job['id']; output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    data, model, args, extract, predict, train0, train1, E0, R0, initial, bound = setup(root, cfg, job, output)
    device = E0.device; gen = torch.Generator().manual_seed(job['seed'] + 71)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=job['learning_rate'])
    valid = [data.triples(s, 'valid') for s in (0, 1)]
    vg = torch.Generator().manual_seed(cfg['validation_seed'])
    valid = [q[torch.randperm(len(q), generator=vg)[:cfg['validation_queries_per_split']]] for q in valid]
    records = []; best = -1.; best_epoch = 0
    epochs = 1 if preflight or job['method'] == 'RandomExpansion' else cfg['epochs']
    for epoch in range(1, epochs+1):
        if args: args.epoch = epoch-1
        model.train(); losses = []
        if job['method'] == 'FullFinetune': epoch_train = torch.cat([train1, train0])
        elif job['method'] == 'Replay':
            sampled = train0[torch.randperm(len(train0), generator=gen)[:len(train1)]]
            epoch_train = torch.cat([train1, sampled])
        else: epoch_train = train1
        if job['method'] != 'RandomExpansion':
            for batchid, ids in enumerate(torch.randperm(len(epoch_train), generator=gen).split(cfg['batch_size'])):
                assert time.monotonic()-start < cfg['budget_seconds_per_job'], 'Registered budget exceeded'
                pos = epoch_train[ids].to(device); h, r, t = pos.unbind(1)
                # Author sampler corrupts either end with equal probability and
                # uses randint(0, num_ent-1), excluding the last ID. Preserve this
                # distribution across native bridges and common training controls.
                draw = torch.randint(11257, (2, len(pos), cfg['negatives']), generator=gen).to(device)
                corrupt_tail = (torch.rand((len(pos), cfg['negatives']), generator=gen) > .5).to(device)
                nh = torch.where(corrupt_tail, h[:, None], draw[0])
                nt = torch.where(corrupt_tail, draw[1], t[:, None])
                nr = r[:, None].expand_as(nh)
                if args:
                    x = torch.cat([pos, torch.stack([nh, nr, nt], -1).reshape(-1, 3)])
                    labels = torch.cat([torch.ones(len(pos), device=device), -torch.ones(nh.numel(), device=device)])
                    loss = model.loss(*x.unbind(1), labels).float()
                else:
                    p = model.triple(h, r, t); n = model.triple(nh, nr, nt)
                    if job['kind'] == 'TransE': loss = F.relu(cfg['margin'] + p - n.mean(1)).mean()
                    else:
                        reg = model.entity(h).square().mean() + model.relation(r).square().mean()
                        loss = F.softplus(-p).mean()+F.softplus(n).mean()+cfg['distmult_l2']*reg
                assert torch.isfinite(loss), 'Nonfinite loss'
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
                if preflight: break
        rec = dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None, train_directed_rows=len(epoch_train))
        if preflight or epoch == 1 or epoch % cfg['validation_every'] == 0 or epoch == epochs:
            model.eval(); E, R = extract()
            with torch.no_grad():
                if args:
                    checkq = valid[1][:8].to(device)
                    assert torch.allclose(predict(checkq), scores(E, R, checkq, job['kind']), atol=1e-12, rtol=1e-12)
            v = [metrics(ranks(E, R, q[:16] if preflight else q, job['kind'])) for q in valid]
            criterion = .5*(v[0]['raw_mrr']+v[1]['raw_mrr']); rec['validation'] = v; rec['selection_score'] = criterion
            if criterion > best:
                best = criterion; best_epoch = epoch
                torch.save(dict(E=E.detach().cpu(), R=R.detach().cpu(), kind=job['kind'], seed=job['seed'],
                                epoch=epoch, native_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                method=job['method'], stage=stage), output/'best.pt')
            print('INCREMENT_PROGRESS', job['id'], epoch, round(criterion, 6), round(time.monotonic()-start, 2), flush=True)
        records.append(rec); (output/'training.json').write_text(json.dumps(records, indent=2))
    bound.update(data.bound)
    for path, digest in bound.items(): assert sha(path) == digest, path
    receipt = dict(id=job['id'], method=job['method'], kind=job['kind'], seed=job['seed'], stage=stage,
                   best_epoch=best_epoch, best_validation_score=best, epochs_executed=epochs,
                   elapsed_s=time.monotonic()-start, parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), test_read=False,
                   bound_input_sha256=bound, checkpoint_sha256=sha(output/'best.pt'),
                   effective_initialization_sha256=__import__('hashlib').sha256(initial.cpu().numpy().tobytes()).hexdigest(),
                   effective_native_options={k:str(v) for k,v in args.__dict__.items()} if args else None)
    (output/'training-receipt.json').write_text(json.dumps(receipt, indent=2))
    torch.save(dict(E=initial.cpu(), R=R0.cpu(), oldE=E0.cpu()), output/'initial.pt')
    print('TRAIN_TERMINAL', job['id'], stage, flush=True)


def evaluate(root, cfg, job):
    torch.set_num_threads(cfg['cpu_threads_per_job']); torch.backends.cuda.matmul.allow_tf32 = False
    output = root/'artifacts/incremental'/job['id']; start = time.monotonic(); device = 'cuda:0'
    checkpoint = torch.load(output/'best.pt', map_location=device, weights_only=True)
    initial = torch.load(output/'initial.pt', map_location=device, weights_only=True)
    data = Data(root, cfg); queries = [data.triples(s, 'test') for s in (0, 1)]
    E, R = checkpoint['E'], checkpoint['R']; kind = job['kind']
    old_before = ranks(initial['oldE'], initial['R'], queries[0], kind)
    old_expanded = ranks(initial['E'], initial['R'], queries[0], kind)
    old_after = ranks(E, R, queries[0], kind); new_after = ranks(E, R, queries[1], kind)
    np.savez_compressed(output/'ranks.npz', old_queries=queries[0].numpy(), new_queries=queries[1].numpy(),
                        old_before=old_before, old_expanded=old_expanded, old_after=old_after, new_after=new_after)
    protected = old_before <= 10; expanded_good = old_expanded <= 10
    summary = dict(id=job['id'], method=job['method'], seed=job['seed'], kind=kind, candidates=11258,
                   old_before=metrics(old_before), old_after=metrics(old_after), new_after=metrics(new_after),
                   old_top10_loss=float((old_after[protected] > 10).mean()), old_top10_protected=int(protected.sum()),
                   same_pool_top10_loss=float((old_after[expanded_good] > 10).mean()),
                   mean_rank_displacement=float((old_after-old_before).mean()),
                   old_parameter_l2=float(torch.linalg.vector_norm(E[:7505]-initial['oldE']).cpu()),
                   relation_parameter_l2=float(torch.linalg.vector_norm(R-initial['R']).cpu()),
                   evaluation_elapsed_s=time.monotonic()-start)
    fixtureq = torch.cat([queries[0][:32], queries[1][:32]])
    with torch.no_grad(): fixtures = torch.cat([scores(E, R, q.to(device), kind).cpu() for q in fixtureq.split(8)]).numpy()
    np.savez_compressed(output/'audit-fixture.npz', queries=fixtureq.numpy(), scores=fixtures)
    (output/'evaluation.json').write_text(json.dumps(summary, indent=2))
    (output/'evaluation-receipt.json').write_text(json.dumps(dict(test_input_sha256=data.bound,
        output_sha256={f.name:sha(f) for f in output.iterdir() if f.is_file()}), indent=2))
    print('EVALUATION_TERMINAL', job['id'], summary, flush=True)
