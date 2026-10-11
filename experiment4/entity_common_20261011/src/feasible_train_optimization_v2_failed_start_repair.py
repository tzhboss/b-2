"""Same TRAIN objective/guards with a bounded feasible-start solver repair.

040 and original active-cut control failure remain unchanged. On a failed
SLSQP cutting-plane solve, retry once from the current row only if that row
satisfies every active pair and absolute ball. No radius/margin/L2 search.
"""
import time
import numpy as np
from feasible_train_optimization import TrainIndex,score_polynomial,TrainLossProblem
from shared_rank_solver_v2 import GuardSeparator


class FeasibleStartProblem(TrainLossProblem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.retry_count=0

    def solve(self,start,options):
        result=super().solve(start,options)
        current=self.E[self.ids].ravel().copy()
        if (not result.success and self.pairs
            and np.min(self.values(current))>=-options['primal_tolerance']
            and np.min(self.ball_values(current))>=-options['movement_tolerance']):
            self.retry_count+=1
            return super().solve(current,options)
        return result


def optimize_train_row(cache,index,entity,center,radius,l2,negative_count,options):
    started=time.monotonic();before=cache.E[[entity]].cpu().numpy().copy()
    positive,negative,keys,scores=index.examples(cache,entity,negative_count)
    problem=FeasibleStartProblem(cache.E.cpu().numpy(),cache.R.cpu().numpy(),entity,
                                 positive,negative,center,radius,l2,options['guard_margin'])
    initial_loss=problem.objective(before.ravel());opt=problem.solve(before,options)
    if not opt.success or not np.isfinite(opt.x).all():
        fallback=cache.propose([entity],before);assert fallback.passed
        return fallback,dict(accepted=False,status='initial_optimizer_failure',optimizer_status=int(opt.status),
             before_loss=initial_loss,after_loss=initial_loss,training_positive_count=len(positive),training_negative_count=len(negative),
             feasible_start_retries=problem.retry_count,history=[],elapsed_s=time.monotonic()-started),positive,negative
    separator=GuardSeparator(cache,[entity],opt.x.reshape(1,-1),[]);history=[];last_status='direct_train_optimum'
    for iteration in range(options['max_cut_rounds']+1):
        rows=opt.x.reshape(1,-1);patch=cache.propose([entity],rows)
        move=float(np.linalg.norm(rows[0]-center));loss=problem.objective(opt.x);bounds=move<=radius+options['movement_tolerance']
        history.append(dict(round=iteration,cuts=len(problem.pairs),violating_queries=int((patch.counts>=cache.k).sum()),
                            optimizer_status=int(opt.status),loss=loss,bounds_ok=bool(bounds),feasible_start_retries=problem.retry_count))
        if patch.passed and bounds and loss<=initial_loss+1e-10:
            return patch,dict(accepted=True,status=last_status,before_loss=initial_loss,after_loss=loss,movements=[move],
                 training_positive_count=len(positive),training_negative_count=len(negative),feasible_start_retries=problem.retry_count,
                 history=history,elapsed_s=time.monotonic()-started,
                 certificate='Complete unchanged ranks, ball and local loss nonincrease; one feasible-start retry after failed SLSQP'),positive,negative
        if iteration==options['max_cut_rounds']:break
        old_size=len(problem.pairs);problem.add(separator.separate(patch,options['cut_rows_per_round']))
        if len(problem.pairs)==old_size:last_status='no_new_separating_rows';break
        opt=problem.solve(rows,options);last_status=f'slsqp_{opt.status}'
        if not opt.success or not np.isfinite(opt.x).all():break
        if np.min(problem.values(opt.x)) < -options['primal_tolerance']:break
    fallback=cache.propose([entity],before);assert fallback.passed
    return fallback,dict(accepted=False,status=last_status,before_loss=initial_loss,after_loss=initial_loss,
         training_positive_count=len(positive),training_negative_count=len(negative),feasible_start_retries=problem.retry_count,
         history=history,elapsed_s=time.monotonic()-started,fallback='Unchanged audited feasible row'),positive,negative
