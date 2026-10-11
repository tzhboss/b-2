"""Prepare/audit official continual KGE sources; never install packages or launch training.
All writes stay in owned repos/{LKGE,IncDE,SAGE}. Registration is controller-owned.
"""
import argparse
import ast
import collections
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

METHODS = ("LKGE", "IncDE", "SAGE")
PRIMARY = {
    "LKGE": [("paper.pdf", "https://ojs.aaai.org/index.php/AAAI/article/download/25539/25311")],
    "IncDE": [("paper.pdf", "https://ojs.aaai.org/index.php/AAAI/article/download/28722/29396")],
    "SAGE": [("paper.pdf", "https://arxiv.org/pdf/2508.11347v1"),
             ("zenodo-release.json", "https://zenodo.org/api/records/15520115")],
}
def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()

def save_json(path, obj):
    raw = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() == raw:
        return
    if path.exists():
        backup = path.with_name(path.name + ".prior-" + str(time.time_ns()))
        backup.write_bytes(path.read_bytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)

def extract_no_overwrite(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("ZIP path escapes owned repository")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            raw = z.read(entry)
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError("Existing file differs from author archive: " + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)

def read_facts(path):
    rows = [tuple(line.split()) for line in path.read_text().splitlines() if line.strip()]
    if any(len(row) != 3 for row in rows):
        raise ValueError("Not H R T text: " + str(path))
    return rows

def audit_dataset(directory):
    by_split = {key: set() for key in ("train", "valid", "test")}
    snapshots = []
    entity_ids, relation_ids = {}, {}
    seen_entities, seen_relations = set(), set()
    files = []
    for i in range(5):
        snapshot = directory / str(i)
        facts = {key: read_facts(snapshot / (key + ".txt")) for key in by_split}
        sets = {key: set(rows) for key, rows in facts.items()}
        for key in sets:
            if len(sets[key]) != len(facts[key]):
                raise ValueError("Duplicate triples " + str(snapshot) + "/" + key)
            if sets[key] & set.union(*by_split.values()):
                raise ValueError("Repeated triple across snapshots")
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if sets[a] & sets[b]:
                raise ValueError("Split overlap")
        for key in by_split:
            by_split[key] |= sets[key]
            path = snapshot / (key + ".txt")
            files.append({"path": str(path.relative_to(directory)), "sha256": sha(path), "bytes": path.stat().st_size})
            for h, r, t in facts[key]:
                seen_entities.update((h, t))
                seen_relations.add(r)
                for e in (h, t):
                    if e not in entity_ids:
                        entity_ids[e] = len(entity_ids)
                if r not in relation_ids:
                    rid = len(relation_ids)
                    relation_ids[r] = rid
                    relation_ids[r + "_inv"] = rid + 1
        extras = []
        for path in sorted(snapshot.glob("*.txt")):
            if path.stem not in by_split:
                extras.append({"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
        snapshots.append({"snapshot": i, **{key: len(v) for key, v in facts.items()},
                          "bidirectional_test_queries": 2 * len(facts["test"]),
                          "cumulative_entities": len(seen_entities),
                          "cumulative_relations_without_inverse": len(seen_relations),
                          "auxiliary": extras})
    totals = {key: len(v) for key, v in by_split.items()}
    return {"dataset": directory.name, "snapshots": snapshots, "totals": totals, "files": files,
            "split_contract": "author-distributed five incremental 3:1:1 splits; not regenerated"}

def declared_defaults(path):
    defaults = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [ast.literal_eval(x) for x in node.args if isinstance(x, ast.Constant)]
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            if flags and "default" in kwargs:
                try: defaults[flags[0]] = ast.literal_eval(kwargs["default"])
                except (ValueError, TypeError): pass
    return defaults

def fetch_primary(repo, method):
    records = []
    for name, url in PRIMARY[method]:
        target = repo / "provenance" / name
        rec = {"url": url, "path": str(target.relative_to(repo))}
        try:
            if not target.exists():
                req = urllib.request.Request(url, headers={"User-Agent": "Research source verification/1.0"})
                with urllib.request.urlopen(req, timeout=45) as response:
                    raw = response.read()
                    rec["resolved_url"] = response.url
                    rec["content_type"] = response.headers.get("content-type")
                if name.endswith(".pdf") and not raw.startswith(b"%PDF"):
                    raise ValueError("Publisher did not return a PDF")
                target.write_bytes(raw)
            rec.update(status="downloaded", sha256=sha(target), bytes=target.stat().st_size)
            if name.endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(target)
                    text = "\n".join((page.extract_text() or "") for page in reader.pages)
                    (target.with_suffix(".txt")).write_text(text)
                    rec["page_count"] = len(reader.pages)
                    rec["repository_or_release_lines"] = [l.strip() for l in text.splitlines()
                         if any(k in l.lower() for k in ("github.com", "zenodo", "source code", "available at"))]
                except Exception as exc:
                    rec["text_extraction_error"] = repr(exc)
        except Exception as exc:
            rec.update(status="unavailable", error=repr(exc))
        records.append(rec)
    return records

def prepare(root, fetch, methods):
    results = []
    for method in methods:
        repo = root / "repos" / method
        provenance = repo / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        archives = [repo / "dataset.zip"] if method == "LKGE" else sorted((repo / "data").glob("*.zip"))
        for archive in archives:
            extract_no_overwrite(archive, repo if method == "LKGE" else repo / "data")
        git = lambda *args: subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
        datasets = [audit_dataset(d) for d in sorted((repo / "data").iterdir()) if d.is_dir() and (d / "0/train.txt").exists()]
        for data in datasets:
            batch = 1024 if method == "IncDE" else 2048
            steps = [(2*s["train"] + batch - 1)//batch for s in data["snapshots"]]
            data["maximum_main_training_optimizer_steps"] = 200 * sum(steps)
            data["optimizer_steps_per_epoch_by_snapshot"] = steps
        source_files = {}
        for rel in git("ls-files").splitlines():
            path = repo / rel
            if path.suffix == ".py" or path.name in ("README.md", "Readme.md", "requirements.txt"):
                source_files[rel] = sha(path)
        result = {"method": method, "commit": git("rev-parse", "HEAD"),
                  "origin": git("remote", "get-url", "origin"),
                  "tag_refs": git("show-ref", "--tags") if method == "SAGE" else None,
                  "defaults": declared_defaults(repo / "src/parse_args.py"),
                  "source_sha256": source_files, "datasets": datasets,
                  "archives": [{"path": str(p.relative_to(repo)), "sha256": sha(p), "bytes": p.stat().st_size} for p in archives],
                  "preparation_only": True, "training_launched": False}
        if fetch:
            result["primary_sources"] = fetch_primary(repo, method)
        save_json(provenance / "preparation_manifest.json", result)
        results.append({"method": method, "commit": result["commit"], "datasets": [d["dataset"] for d in datasets],
                        "steps_ENTITY": next((d["maximum_main_training_optimizer_steps"] for d in datasets if d["dataset"]=="ENTITY"), None),
                        "primary": result.get("primary_sources")})
    print(json.dumps(results, indent=2))

def audit_ranks(root, directory, method, dataset, split_profile):
    import numpy as np
    repo = root / "repos" / method
    entity_ids, relation_ids = {}, {}
    all_truth, tests, candidates = set(), [], []
    for i in range(5):
        facts = {key: read_facts(repo / "data" / dataset / str(i) / (key + ".txt")) for key in ("train","valid","test")}
        if method == "LKGE" and split_profile == "native":
            facts["valid"] = facts["test"]
        for key in ("train","valid","test"):
            for h,r,t in facts[key]:
                for e in (h,t):
                    if e not in entity_ids: entity_ids[e]=len(entity_ids)
                if r not in relation_ids:
                    rid=len(relation_ids);relation_ids[r]=rid;relation_ids[r+"_inv"]=rid+1
        for key in facts:
            for h,r,t in facts[key]:
                hh,rr,tt=entity_ids[h],relation_ids[r],entity_ids[t]
                all_truth.add((hh,rr,tt));all_truth.add((tt,rr+1,hh))
        forward=sorted([(entity_ids[h],relation_ids[r],entity_ids[t]) for h,r,t in facts["test"]],key=lambda row:row[1])
        bidirectional=forward+[(t,r+1,h) for h,r,t in forward]
        filters=collections.defaultdict(set)
        for h,r,t in all_truth: filters[(h,r)].add(t)
        tests.append((bidirectional,filters));candidates.append(len(entity_ids))
    records=[]
    expected={f"train{i}_test{j}_post_train.npz" for i in range(5) for j in range(i+1)}
    expected|={f"train{i}_test{i}_pre_fwt.npz" for i in range(1,5)}
    if method == "SAGE":
        expected |= {f"train{i}_test{j}_post_expand.npz" for i in range(4) for j in range(i+1)}
    missing=sorted(expected-{p.name for p in directory.glob("*.npz")})
    for path in sorted(directory.glob("*.npz")):
        with np.load(path,allow_pickle=False) as a:
            i,j=int(a["trained_snapshot"]),int(a["tested_snapshot"])
            triples=np.column_stack([a[k] for k in ("head","relation","tail")])
            ranks=a["rank"]
            truth,filters=tests[j]
            assert np.array_equal(triples,np.asarray(truth)),(path.name,"query IDs/order/count")
            assert ranks.dtype.kind in "iu" and np.all(ranks>=1) and np.all(ranks<=candidates[j])
            assert np.all(a["candidates"]==candidates[j])
            assert np.array_equal(a["filtered_count"],[len(filters[(h,r)]) for h,r,t in truth])
            metrics={"mrr":float(np.mean(1.0/ranks.astype(np.float64)))}
            metrics.update({f"hits{k}":float(np.mean(ranks<=k)) for k in (1,3,5,10)})
            records.append({"file":path.name,"trained_snapshot":i,"tested_snapshot":j,"queries":len(ranks),
                            "sha256":sha(path),"raw_metrics":metrics,
                            "author_rounded4":{key:round(val,4) for key,val in metrics.items()}})
    aggregates=[]
    for i in range(5):
        rows=[r for r in records if r["trained_snapshot"]==i and r["file"].endswith("post_train.npz")]
        if len(rows)!=i+1:continue
        denominator=sum(r["queries"] for r in rows)
        exact={k:sum(r["raw_metrics"][k]*r["queries"] for r in rows)/denominator for k in ("mrr","hits1","hits3","hits10")}
        author={k:round(sum(r["author_rounded4"][k]*r["queries"] for r in rows)/denominator,3) for k in exact}
        aggregates.append({"trained_snapshot":i,"query_weighted_exact":exact,"author_round4_then_weight_then_round3":author})
    print(json.dumps({"complete":not missing,"missing":missing,"files":records,"aggregates":aggregates,
                     "boundary":"Independent metric aggregation and query/filter contract checks. Does not by itself independently verify model scores."},indent=2))
    if missing:sys.exit(2)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--fetch-primary",action="store_true")
    ap.add_argument("--audit-ranks",type=Path)
    ap.add_argument("--method",choices=METHODS,default="IncDE")
    ap.add_argument("--dataset",default="ENTITY")
    ap.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    ap.add_argument("--split-profile",choices=("native","clean"),default="native")
    args=ap.parse_args()
    if args.audit_ranks:audit_ranks(args.root.resolve(),args.audit_ranks,args.method,args.dataset,args.split_profile)
    else:prepare(args.root.resolve(),args.fetch_primary,args.methods)
if __name__=="__main__":main()
