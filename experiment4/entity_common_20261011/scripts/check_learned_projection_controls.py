"""Independent shared-model Top10 example for TRUE/Prediction interactions."""
import argparse,json,sys
from pathlib import Path
import numpy as np


def run(root,config):
    sys.path.insert(0,str(root/'src'));from learned_rank_projection import frontier_cpu
    from safe_proposal import nearest_feasible,verify_projection,radius_segment
    cfg=json.loads((root/config).read_text());opts=cfg['solver']
    old=np.column_stack([np.arange(12,0,-1),np.r_[np.arange(1,10),0,-1,-2]]).astype(float)
    E0=np.vstack([old,[-1.,-1.],[-.5,-2.]])
    R=np.array([[1/12,0.],[0.,1.]])
    G=(E0[0]*R[0])[None,:];P=(E0[0]*R[1])[None,:]
    tg=np.array([G[0]@E0[9]]);pg=np.array([P[0]@E0[9]])
    def actual_ranks(E):
        values=[]
        for relation in (0,1):
            s=(E[0]*R[relation])@E.T;values.append(int(1+(s>s[9]).sum()))
        return values
    assert actual_ranks(E0)==[10,10]
    target=np.array([[8.,5.],[7.,4.]]);rows=[]
    for protection in ['none','true','pred']:
        E=E0.copy();tc=np.array([9]);pc=np.array([9]);good_t=tc<=9;good_p=pc<=9;checked=0
        for tail,goal in zip([12,13],target):
            before=E[tail].copy();u=goal-before
            A,b=frontier_cpu(G,P,tg,pg,before,tc,pc,good_t,good_p,protection,cfg['guard_margin'])
            d,c=nearest_feasible(u,A,b,opts);assert c['certified'];verify_projection(u,A,b,d,c,opts)
            alpha=radius_segment(before,d,float(np.linalg.norm(E0,axis=1).max()));E[tail]=before+alpha*d
            nt=tc+(G@E[tail]>tg).astype(int)-(G@before>tg).astype(int)
            np_=pc+(P@E[tail]>pg).astype(int)-(P@before>pg).astype(int)
            # Recompute actual full-candidate ranks, not saved tracking arrays.
            actual=actual_ranks(E)
            assert actual==[int(1+nt[0]),int(1+np_[0])]
            if protection!='none':assert actual[0]<=10
            if protection=='pred':assert actual[1]<=10
            tc=nt;pc=np_;checked+=1
        assert np.array_equal(E[:12],old)
        if protection=='none':assert actual_ranks(E)==[12,12]
        if protection=='true':assert actual_ranks(E)==[10,12]
        if protection=='pred':assert actual_ranks(E)==[10,10]
        before_head=(E0[12]*R[1])@E0.T;after_head=(E[12]*R[1])@E.T
        assert not np.array_equal(before_head,after_head),'New head must use the updated shared vector'
        rows.append(dict(protection=protection,steps_checked=checked,actual_final_old_ranks=actual_ranks(E),new_head_scores_changed=True,old_parameters_unchanged=True))
    # A nonzero proposal into a contradictory zero-slack wedge has certified
    # optimum zero; this case must be reported as no learning, not improvement.
    d,c=nearest_feasible(np.array([1.,1.]),np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),np.zeros(4),opts)
    assert c['certified'] and np.linalg.norm(d)<1e-12
    out=root/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False)
    result=dict(check_scope='Synthetic shared old/new vector model; no real graph or utility evidence',passed=True,
                cases=rows,conflict_optimum_is_zero=True,graph_facts_read=False,test_read=False,
                limitations=['Demonstrates registered scope and utility conflict only; cannot establish graph performance.'])
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();run(a.root.resolve(),a.config)
