# COLING 2027 / ARR format notes

This draft targets a COLING 2027 long-paper submission through ACL Rolling Review (ARR).

Current official requirements checked on 2026-09-22:
- COLING 2027 main-conference papers must be submitted through ARR.
- Latest ARR cycle for COLING 2027: 2026-10-12.
- Long paper: up to 8 pages of main content.
- The Limitations section is mandatory, after the conclusion and before references, and does not count toward the main-content page limit.
- References are unlimited; appendix is unlimited but the main paper must be self-contained.
- Review is anonymized; the review PDF must contain no identifying author information.
- COLING 2027 explicitly directs authors to the ARR guidelines and ARR templates.
- ARR discontinued Word submissions for conference review starting in March 2026, so this project uses the official ACL/ARR LaTeX style.

Official pages:
- COLING 2027 main CFP: https://2027.coling-iccl.org/calls/main_conference_papers/
- ARR CFP: https://aclrollingreview.org/cfp
- ARR author guidelines: https://aclrollingreview.org/authors
- ACL formatting/template repository: https://github.com/acl-org/acl-style-files

Files:
- main.tex: anonymous review draft.
- acl.sty / acl_natbib.bst: downloaded from the official ACL style repository.
- references.bib: working bibliography.
- ACL_FORMATTING.md: upstream formatting instructions copied from the official template repository.
- figures/: paper figures copied from audited experiment assets.

Compile (when TeX is available):
pdflatex main
bibtex main
pdflatex main
pdflatex main
