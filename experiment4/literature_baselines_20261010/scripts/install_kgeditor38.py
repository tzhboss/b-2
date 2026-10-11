import pathlib,subprocess,time
R=pathlib.Path(__file__).resolve().parents[1]
base=R/'environment/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8'
env=R/'environment/kgeditor38';assert not env.exists()
subprocess.run([str(base),'-m','venv',str(env)],check=True)
p=str(env/'bin/python')
commands=[[p,'-m','pip','install','pip==23.3.2','setuptools<70','wheel'],[p,'-m','pip','install','torch==1.12.0+cu113','torchvision==0.13.0+cu113','--extra-index-url','https://download.pytorch.org/whl/cu113'],[p,'-m','pip','install','numpy==1.23.5','transformers==4.20.1','pytorch-lightning==1.3.1','torchmetrics==0.5.1','allennlp==2.10.1','higher==0.2.1','tensorboardX','protobuf==3.20.3']]
for cmd in commands:
 print('INSTALL_STEP',cmd[4:],flush=True);subprocess.run(cmd,check=True,timeout=1200)
print('INSTALL_COMPLETE',flush=True)
