import pathlib,subprocess
R=pathlib.Path(__file__).resolve().parents[1];p=str(R/'environment/kgeditor38/bin/python')
constraint=R/'environment/kgeditor38-wheel-constraints.txt'
constraint.write_text('cymem==2.0.7\nmurmurhash==1.0.9\npreshed==3.0.8\nspacy==3.3.3\nthinc==8.0.17\nnumpy==1.23.5\nPyYAML==5.4.1\nlmdb==1.4.1\n')
cmd=[p,'-m','pip','install','--only-binary=:all:','--no-binary=fairscale,termcolor,pathtools','-c',str(constraint),'numpy==1.23.5','transformers==4.20.1','pytorch-lightning==1.3.1','torchmetrics==0.5.1','allennlp==2.10.1','higher==0.2.1','tensorboardX','protobuf==3.20.3']
subprocess.run(cmd,check=True,timeout=1200)
print('ENV_REPAIR_COMPLETE',flush=True)
