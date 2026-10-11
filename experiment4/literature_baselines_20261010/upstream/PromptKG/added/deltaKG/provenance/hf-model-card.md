---
datasets:
- ChancesYuan/KGEditor
language:
- en
pipeline_tag: token-classification
---

# Model description
We propose a task that aims to enable data-efficient and fast updates to KG embeddings without damaging the performance of the rest. 
We provide four experimental edit object models of the PT-KGE in the paper experiments used.

### How to use

Here is how to use this model:

```python
>>> from transformers import BertForMaskedLM
>>> model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path="zjunlp/KGEditor", subfolder="E-FB15k237")
```

### BibTeX entry and citation info
```bibtex
@article{DBLP:journals/corr/abs-2301-10405,
  author    = {Siyuan Cheng and
               Ningyu Zhang and
               Bozhong Tian and
               Zelin Dai and
               Feiyu Xiong and
               Wei Guo and
               Huajun Chen},
  title     = {Editing Language Model-based Knowledge Graph Embeddings},
  journal   = {CoRR},
  volume    = {abs/2301.10405},
  year      = {2023},
  url       = {https://doi.org/10.48550/arXiv.2301.10405},
  doi       = {10.48550/arXiv.2301.10405},
  eprinttype = {arXiv},
  eprint    = {2301.10405},
  timestamp = {Thu, 26 Jan 2023 17:49:16 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2301-10405.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
