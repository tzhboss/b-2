"""Exact TRAIN-only node centrality, partitioned over every Brandes source."""
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import numpy as np
from common_incremental import Data
from common_training import sha

_graph=None
def init_graph(graph):
    global _graph
    _graph=graph

def part_centrality(sources):
    # Every target and every source is included; no stochastic k approximation.
    return nx.betweenness_centrality_subset(_graph,sources,list(_graph),normalized=True)

def feature_precheck():
    errors=[]
    for graph in [nx.path_graph(9),nx.cycle_graph(9),nx.star_graph(8),nx.disjoint_union(nx.path_graph(6),nx.path_graph(4)),nx.gnp_random_graph(17,.3,seed=12)]:
        init_graph(graph);nodes=list(graph)
        pieces=[part_centrality(nodes[i::3]) for i in range(3)]
        actual={v:sum(p[v] for p in pieces) for v in nodes};expected=nx.betweenness_centrality(graph)
        error=max(abs(actual[v]-expected[v]) for v in nodes);errors.append(error);assert error<1e-14
    return errors

def generate(root,cfg):
    started=time.monotonic();data=Data(root,cfg);train=data.triples(1,'train')[::2].tolist()
    graph=nx.Graph()
    for h,r,t in train:graph.add_edge(h,t)
    errors=feature_precheck();nodes=list(graph);groups=[nodes[i::cfg['feature_partitions']] for i in range(cfg['feature_partitions'])]
    assert sorted(x for group in groups for x in group)==sorted(nodes)
    folder=root/cfg['feature_directory'];folder.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=cfg['feature_workers'],initializer=init_graph,initargs=(graph,)) as pool:
        parts=list(pool.map(part_centrality,groups))
    centrality={v:sum(part[v] for part in parts) for v in nodes};degree=nx.degree_centrality(graph)
    for name,values in [('train_nodes_betweenness.txt',centrality),('train_nodes_degree.txt',degree)]:
        (folder/name).write_text(''.join(f'{v}\t{value}\n' for v,value in values.items()))
    receipt=dict(status='PASS_PARTITIONED_EXACT_TRAIN_ONLY',graph_nodes=len(nodes),graph_edges=graph.number_of_edges(),
                 all_sources_covered=len(nodes),all_targets_covered=len(nodes),approximate=False,
                 precheck_max_errors=errors,elapsed_s=time.monotonic()-started,networkx=nx.__version__,
                 input_sha256=data.bound,source_sha256=sha(Path(__file__)),
                 output_sha256={f.name:sha(f) for f in folder.iterdir()},
                 limits=['Partitioned summation may differ in floating-point last bits from single-loop NetworkX; native weights remain exact full-source centrality.'])
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
