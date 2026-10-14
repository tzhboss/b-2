import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
sys.path.insert(0,'/data/lc/untzh/init_protect123_20261014')
import experiment123 as runner
runner.ROOT=Path(__file__).resolve().parent
runner.INITS=['zero','random_half']
if __name__=='__main__':
    try:
        runner.base.write(runner.ROOT/'driver.json',dict(implementation='experiment123.py',inits=runner.INITS,stdout_tag='EVALUATED123 is reused implementation tag; directory and metadata identify126'))
        runner.main()
    except Exception as exc:runner.status('failed',error=repr(exc));raise
