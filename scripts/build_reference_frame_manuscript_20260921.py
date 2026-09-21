from pathlib import Path
root=Path('/data/lc/tzh/paper_draft')
parts=[
'00_abstract.md',
'01_introduction.md',
'02_related_work.md',
'03_method.md',
'04_results.md',
'05_discussion.md',
'06_conclusion.md',
]
out=root/'manuscript.md'
with out.open('w') as f:
    f.write('# Prosodic Reference Frames: Draft Manuscript\n\n')
    for i,p in enumerate(parts):
        txt=(root/p).read_text().strip()
        f.write(txt+'\n\n')
    f.write('---\n\n')
    f.write('## Draft Reference Notes\n\n')
    f.write((root/'05_reference_notes.md').read_text().replace('# Reference Notes','').strip()+'\n')
print(out)
