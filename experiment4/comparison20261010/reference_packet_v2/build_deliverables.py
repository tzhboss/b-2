"""Create source-preserving review PDF and a separately grounded comparison."""
from pathlib import Path
import csv, json, hashlib, re, html, shutil
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
import fitz

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
KGE=WORK/'rank-locality-followup30'
LIT=Path('D:/桌面/webcodex/claude-literature-review/research/dasfaa2027')
OUT=WORK/'output/pdf'
TMP=WORK/'tmp/pdfs/kge20261010'
OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readj(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def gather():
    replay=readj(ROOT/'remote-evidence/replay_manifest.json');assert replay['status']=='PASS'
    for item in replay['outputs']:
        assert sha(ROOT/'remote-evidence'/item['file'])==item['sha256']
    audit34=readj(KGE/'results/KGE-RANK-034/audit.json')
    audit36=readj(KGE/'results/KGE-RANK-036/audit.json')
    assert audit34['status']==audit36['status']=='PASS'
    audits37=[readj(KGE/'artifacts'/f'{kind}-all-support400-audit/audit.json') for kind in ('distmult','complex')]
    assert all(a['status']=='PASS' and a['n_cases']==400 and a['n_audited_arms']==800 for a in audits37)
    canonical37=readj(KGE/'experiments/KGE-RANK-037/experiment.json')
    promoted37=readj(KGE/'results/KGE-RANK-037/audit.json')
    assert canonical37['status']=='completed' and canonical37['active_processes']==[]
    assert canonical37['audit']['status']==promoted37['status']=='PASS'
    assert canonical37['audit']['checks']==promoted37['checks']==sum(a['checks'] for a in audits37)==49360
    for a in audits37:
        for name,digest in a['input_artifact_sha256'].items():assert sha(KGE/name)==digest
    prior=readj(LIT/'reproduction_rank_locality/final_reproduction_audit.json')
    remote=readcsv(ROOT/'remote-evidence/phase24_v2_rank_kernel_audit.csv')
    summary34=[r for r in readcsv(KGE/'results/KGE-RANK-034/summary.csv') if r['seed']=='all']
    summary36=[r for r in readcsv(KGE/'results/KGE-RANK-036/summary.csv') if r['seed']=='all']
    rows=[]
    for a in audits37:
        for r in a['summaries']:
            if r['seed']=='all':rows.append(r)
    return replay,audit34,audit36,audits37,prior,remote,summary34,summary36,rows

def make_report(data):
    replay,a34,a36,a37,prior,remote,s34,s36,s37=data
    lines=['# 知识图谱嵌入编辑：远程复核与文献基线实测对照',
           '**2026-10-10 · experiment4 · 结果复用及独立审计**',
           '## 1. 结论与本次完成事项',
           '可以复用已有数据、权重和结果做有边界的对照。当前已有 Rank-Level Locality 的部分复现、同400案例的QP比较，以及远程64维共享参数三目标实验。它们对应不同协议，下面分别报告。30篇综述条目并非30个同任务可替换编辑器；本次没有把它们全部重训或拼成统一SOTA表。',
           '本次在 /data/lc/tzh/experiment4/comparison20261010 新建隔离目录，重放第24轮保存结果的哈希与汇总审计。30个checkpoint哈希、18个编辑文件核对通过，32行汇总成功重建；三个审计/汇总文件与旧版逐字节一致，18个原编辑文件未改变。没有新增训练、模型推理或官方TEST数据。',
           f'另对本地037两模型共1600个已保存Support更新执行独立SVD、支持行选择、完整排名及字节绑定审核，共{sum(a["checks"] for a in a37):,}项记录检查，0差异。数值正确性PASS不代表方法性能通过，也不是独立科研评审或原创性认证。',
           '## 2. 原论文数值与公开权重部分复现',
           '论文来源：https://arxiv.org/html/2609.12116v1；公开代码：https://github.com/pa0lai/kge-rank-locality 。下面读取既有final_reproduction_audit.json，不是本次新跑1000个TEST案例。每模型重建1000个案例；作者原逐例清单不可得，两个队列不同，差值不能当作配对胜负。历史TEST已暴露。',
           '| 模型 | 方法 | 论文SafeSucc | 重建队列SafeSucc |',
           '|---|---|---|---|']
    allrows=[]
    for kind,r in prior['results'].items():
        for m in r['methods']:
            own=m['ours_reconstructed_case_cohort'];paper=m['published_original_paper_different_cohort']
            paper_safe=f'{paper["Safe"]*100:.2f}%' if paper is not None else '未提供主表对应值'
            lines.append(f'| {kind} | {m["method"]} | {paper_safe} | {own["Safe"]*100:.2f}% |')
            allrows.append(dict(panel='historical_reconstructed_TEST1000',model=kind,method=m['method'],N=1000,success=own['S'],safe_success=own['Safe'],damage=own['D'],external_damage='',norm='',paper_safe=paper['Safe'] if paper else '',scope='Different author cohort; historical exposed TEST; partial reproduction'))
    lines+=['Support主表原论文SafeSucc为DistMult36.3%、ComplEx37.7%；它不是此重建TEST表已精确复现的方法。',
            '## 3. 同权重、同400个VALID请求的固定比较',
            '034的两个模型使用公开256维权重，单候选尾列更新；原head查询向量冻结。所有400次包括拒绝和Rank回退。SafeSucc要求目标进入raw Top10且规定范围内的TRAIN原正确事实无一掉榜。VALID已暴露，不能称最终盲测。',
            '| 模型 | 方法 | 成功率 | SafeSucc | 外部损伤/次 |',
            '|---|---|---|---|---|']
    labels={'Direct':'Direct','PublishedRank':'Rank(.125)','SupportRegularized_TRAIN_adaptation':'Support TRAIN适配','PredictionSetQP_RankNormCap':'TRUE+Prediction QP及Rank回退','TrueOnlyQP_RankNormCap':'TRUE-only QP及Rank回退'}
    for r in s34:
        lines.append(f'| {r["model"]} | {labels.get(r["method"],r["method"])} | {float(r["success_at_k"])*100:.2f}% | {float(r["safe_success"])*100:.2f}% | {float(r["external_valid_damage"]):.4f} |')
        allrows.append(dict(panel='034_same_VALID400',model=r['model'],method=r['method'],N=r['N'],success=r['success_at_k'],safe_success=r['safe_success'],damage=r['damage_count'],external_damage=r['external_valid_damage'],norm=r['update_norm'],paper_safe='',scope='TRAIN-input deployment; single candidate column; exposed VALID'))
    lines+=['QP相对Rank的SafeSucc提升9.50/10.75个百分点，结论限定于这个固定队列。TRAIN支持适配不是原作者all-split Support，不能由此称战胜原论文最强完整系统。',
            '## 4. 036机制消融：收益归属必须修正',
            '| 模型 | 方法 | SafeSucc | 外部损伤/次 |',
            '|---|---|---|---|']
    for r in s36:
        lines.append(f'| {r["model"]} | {labels.get(r["method"],r["method"])} | {float(r["safe_success"])*100:.2f}% | {float(r["external_valid_damage"]):.4f} |')
        allrows.append(dict(panel='036_same_VALID400_ablation',model=r['model'],method=r['method'],N=r['N'],success=r['success_at_k'],safe_success=r['safe_success'],damage=r['damage_count'],external_damage=r['external_valid_damage'],norm=r['update_norm'],paper_safe='',scope='Same exposed034; reused rows are not additional independent cases'))
    lines+=['加入预测集合约束的额外SafeSucc贡献未通过固定门槛：DistMult -0.25个百分点，ComplEx 0.00个百分点。其主要支持证据是减少已覆盖查询的旧事实损伤。没有TRAIN查询键的外部事实桶中，Rank原本损失为0；当前低事件数不能支持未知查询泛化改善。',
            '本次复用了现有TRUE-only与TRUE+Prediction两臂。Prediction-only QP尚未实测；不能称完整三臂消融已经完成。',
            '## 5. 037 all-split Support：声明约定的重建',
            '冻结同034400请求，按原论文软目标 min(q^Tδ-Δ)^2+||Cδ||²，最多256个最高原始范数same-tail支持行。原作者目标排除、重复/并列处理、数值内核与原队列未完全公开，因此保留两种事先声明的约定，不挑表现较好的版本冒充原作。',
            '| 模型 | 目标处理约定 | 成功率 | SafeSucc | 外部损伤/次 |',
            '|---|---|---|---|---|']
    for r in s37:
        label='目标保持可选' if 'Literal' in r['method'] else '排除全部请求事实'
        lines.append(f'| {r["model"]} | {label} | {r["success_at_k"]*100:.2f}% | {r["safe_success"]*100:.2f}% | {r["external_valid_damage"]:.4f} |')
        allrows.append(dict(panel='037_same_VALID400_retrospective',model=r['model'],method=r['method'],N=r['N'],success=r['success_at_k'],safe_success=r['safe_success'],damage=r['damage_count'],external_damage=r['external_valid_damage'],norm=r['update_norm'],paper_safe='',scope='TRAIN+VALID+TEST support labels; author target policy unknown; reconstruction'))
    lines+=['在这个队列，软目标的目标处理约定显著影响结果。目标本身被支持项惩罚时，提升与保护要求可能冲突；这不证明原论文实现错误。排除请求事实的重建SafeSucc为54.00%/53.25%，低于固定QP的57.75%/59.75%，属于同队列、不同实际信息输入的描述性比较，不是精确原作者最强方法战胜或公平部署结论。',
            '037显式读取TEST事实标签作为优化支持输入，没有新计算TEST评价分数/排名。外部VALID标签也可进入支持矩阵，因此其外部损伤不能当作整个未参与优化的留出评价。',
            '## 6. 远程 experiment4：64维共享Head/Tail三目标',
            '远程六个模型：DistMult/ComplEx各3训练种子401/402/403，25epochs，64维；每种81次尝试，B=5.2，成本为逐实体L2范数之和。三个目标可能涉及不同实体，不等于同一tail向量的三请求联合编辑。该表使用跨关系TRAIN-heldout损伤，不是034的SafeSucc。',
            '| 模型 | 方法 | 三目标成功 | 跨关系损伤/次 | L2,1/次 |',
            '|---|---|---|---|---|']
    for r in remote:
        if float(r['budget'])!=5.2:continue
        lines.append(f'| {r["kind"]} | {r["method"]} | {r["n_success"]}/81 | {float(r["cross_damage_per_attempt"]):.3f} | {float(r["L21_per_attempt"]):.3f} |')
        allrows.append(dict(panel='remote24_shared3target64d_B5.2',model=r['kind'],method=r['method'],N=r['attempts'],success=r['success_rate'],safe_success='',damage=r['cross_damage_per_attempt'],external_damage='',norm=r['L21_per_attempt'],paper_safe='',scope='Shared full entity table; three targets; L2,1; different trained weights'))
    lines+=['LocalGuard相对Direct跨关系损伤差的查询聚类探索性95%区间：DistMult [-0.2564, 0.0128]；ComplEx [-0.0897, 0.0385]，均含零。此协议下尚未支持显著超越Direct。作者Rank数值内核用于本地多目标扩展，不能用此结果否定原论文完整方法。',
            '## 7. 30篇综述条目的复现路由',
            'method_routing.csv逐条保留原综述的标题、来源、代码链接和协议分类，并增加本轮可比性判断。链接“存在于原综述”不等于本次验证所有仓库可运行。',
            '- Rank-Level Locality：优先直接对照；已有Direct/Rank/StrictNull/Ridge部分复现，Support采用显式约定重建。',
            '- KGEditor与KBS2026：LM-based KGE，需其架构、EDIT/ADD数据、训练编辑器及相同rank-damage指标，不能把本文SafeSucc与其论文数字硬比。',
            '- AIR、IncDE、FastKGE、SIT、SAGE、DebiasedKGE、STARK等：多快照动态/持续KGE，先建立快照、实体增长、MRR/Hits/遗忘和成本协议，再谈复现。',
            '- GEM、OGD、AlphaEdit等：约束思想与创新性审计前作，不能直接作为浅层KGE同任务SOTA数值。',
            '本次只直接网页核对Rank论文/仓库与KGEditor出版页；其余30条元数据继承原范围综述，不能称本次全部独立全文核验。',
            '## 8. 去留判断与下一步',
            'GO：形成有出处、分协议、可复核的实验对照材料。尚无全SOTA战胜、新算法优先权或投稿GO。037正确性门槛通过，性能为描述性结果；目标处理差异应作为复现限制保留。',
            '若继续方法研发，先补Prediction-only机制臂和更严格的未知查询损伤评价；再单独注册一个共同tail向量的三请求任务，并对实际共享Head/Tail实体表重新评分。此前的冻结尾列证书不能自动扩展为全共享模型安全保证。当前不扩大模型、参数网格或新的TEST评价来追求正结果。',
            '## 9. 可复核文件与执行边界',
            f'本地对照目录：{ROOT.as_posix()}。远程重放目录：/data/lc/tzh/experiment4/comparison20261010 。remote-evidence/replay_manifest.json记录逐字节相同及原始文件不变；comparison_results.csv提供所有分面结果，method_routing.csv提供30篇路由。',
            '037审计原件保存在rank-locality-followup30/artifacts/{distmult,complex}-all-support400-audit/audit.json；原034/036结果与失败历史保留。本报告生成器和manifest.json记录所消费文件SHA-256，PDF原综述转换版保留原文字，不把本轮新结果静默写入原综述。']
    text='\n\n'.join(lines)
    # Remove blank lines within consecutive Markdown table rows.
    text=re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n',text)
    (ROOT/'comparison_report.md').write_text(text+'\n',encoding='utf8')
    writecsv(ROOT/'comparison_results.csv',allrows)
    routes=readcsv(LIT/'kge_editing_20261010_papers.csv')
    for r in routes:
        cat=r['category']
        if r['id']=='rank-locality-2026': status='PARTIAL_REPRODUCTION_AND_FIXED_SAME_COHORT_COMPARISON';why='Direct/Rank/StrictNull/Ridge available; Support reconstruction; exact author cohort/kernel incomplete'
        elif cat.startswith('B'):status='SEPARATE_LM_KGE_PROTOCOL_REQUIRED';why='Architecture/training and EDIT/ADD differ; no shallow KGE percentage comparison'
        elif cat.startswith('C'):status='SEPARATE_CONTINUAL_KG_PROTOCOL_REQUIRED';why='Snapshots/entity growth/MRR/Hits/forgetting differ'
        else:status='RELATED_MECHANISM_OR_SOURCE_ONLY';why='Theory/LLM/unlearning differs; no same-task score produced'
        r.update(current_disposition=status,reason=why,current_run=False,metadata_status='Inherited scoped review; not all independently reverified in this task')
    assert len(routes)==30
    writecsv(ROOT/'method_routing.csv',routes)

pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simhei.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCN',fontName='CN',fontSize=9.4,leading=15.0,spaceAfter=7,wordWrap='CJK',textColor=colors.HexColor('#25364B')))
styles.add(ParagraphStyle(name='TitleCN',parent=styles['BodyCN'],fontSize=20,leading=27,spaceAfter=16,textColor=colors.HexColor('#133F69')))
styles.add(ParagraphStyle(name='HeadCN',parent=styles['BodyCN'],fontSize=13,leading=19,spaceBefore=13,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#133F69')))
styles.add(ParagraphStyle(name='SubCN',parent=styles['HeadCN'],fontSize=10.8,leading=16))
styles.add(ParagraphStyle(name='CellCN',parent=styles['BodyCN'],fontSize=8.1,leading=12,spaceAfter=0))

def inline(s):
    s=s.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2212','-').replace('²','^2')
    s=html.escape(s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<link href="{m[2]}" color="#216998">{m[1]}</link>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font color="#496A88">\1</font>',s)
    return s

def make_pdf(md,target,label):
    story=[];lines=md.read_text(encoding='utf-8-sig').splitlines();i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s:i+=1;continue
        if s.startswith('|'):
            raw=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c or '-') for c in cells):raw.append(cells)
                i+=1
            n=max(map(len,raw));data=[[Paragraph(inline(c),styles['CellCN']) for c in row+['']*(n-len(row))] for row in raw]
            # Wide method names and source descriptions need proportionate space.
            weights=([.14,.30,.28,.28] if n==4 and label=='review' else [.14,.40,.15,.15,.16] if n==5 else [1/n]*n)
            if len(weights)!=n:weights=[1/n]*n
            t=Table(data,colWidths=[487*w for w in weights],repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3EDF5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#527D9D')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F8FA')]),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#DEE5EC'))]))
            story += [t,Spacer(1,9)];continue
        match=re.match(r'^(#{1,6})\s+(.*)',s)
        if match:
            style='TitleCN' if len(match[1])==1 else 'HeadCN' if len(match[1])==2 else 'SubCN'
            story.append(Paragraph(inline(match[2]),styles[style]));i+=1;continue
        block=[s];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||- )',lines[i].strip()):
            block.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(block)),styles['BodyCN']))
    def footer(c,d):
        c.setStrokeColor(colors.HexColor('#D4DFE9'));c.line(54,43,541,43)
        c.setFont('CN',8);c.setFillColor(colors.HexColor('#62798D'))
        c.drawString(54,30,'KGE editing / 2026-10-10');c.drawRightString(541,30,str(d.page))
    SimpleDocTemplate(str(target),pagesize=(595.28,841.89),rightMargin=54,leftMargin=54,topMargin=48,bottomMargin=58,title=md.stem,author='Research comparison').build(story,onFirstPage=footer,onLaterPages=footer)
    doc=fitz.open(target)
    texts=[p.get_text() for p in doc]
    assert len(doc)>0 and all(len(t.strip())>30 for t in texts)
    assert not any('\ufffd' in t or '\u25a0' in t for t in texts)
    for idx,page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.0,1.0)).save(TMP/f'{label}-{idx+1:02d}.png')
    # Every page at once for visual review; separate images retain readable detail.
    from PIL import Image,ImageOps,ImageDraw
    files=[TMP/f'{label}-{idx+1:02d}.png' for idx in range(len(doc))]
    thumbs=[]
    for p in files:
        im=Image.open(p).convert('RGB');im.thumbnail((298,421));thumbs.append(im)
    cols=3;rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new('RGB',(cols*318,rows*451),'#d9e1e9')
    for idx,im in enumerate(thumbs):sheet.paste(im,((idx%cols)*318+10,(idx//cols)*451+10))
    sheet.save(TMP/f'{label}-contact.png')
    return {'pages':len(doc),'text_chars':sum(map(len,texts)),'sha256':sha(target),'all_pages_rendered':True}

if __name__=='__main__':
    data=gather();make_report(data)
    outputs={}
    for source,name,label in [(LIT/'review_kge_editing_2026_10_10.md','kge_editing_literature_20261010.pdf','review'),(ROOT/'comparison_report.md','kge_remote_comparison_20261010.pdf','comparison')]:
        outputs[name]=make_pdf(source,OUT/name,label)
    sources=[LIT/'review_kge_editing_2026_10_10.md',LIT/'kge_editing_20261010_papers.csv',LIT/'reproduction_rank_locality/final_reproduction_audit.json',KGE/'results/KGE-RANK-034/audit.json',KGE/'results/KGE-RANK-034/summary.csv',KGE/'results/KGE-RANK-036/audit.json',KGE/'results/KGE-RANK-036/summary.csv',ROOT/'remote-evidence/replay_manifest.json',ROOT/'remote-evidence/phase24_v2_rank_kernel_audit.csv',ROOT/'comparison_report.md',Path(__file__)]
    sources += [KGE/'artifacts'/f'{k}-all-support400-audit/audit.json' for k in ('distmult','complex')]
    sources += [KGE/'results/KGE-RANK-037/audit.json',KGE/'experiments/KGE-RANK-037/experiment.json']
    (ROOT/'manifest.json').write_text(json.dumps({'date':'2026-10-10','outputs':outputs,'source_sha256':{str(p.resolve()):sha(p) for p in sources},'scope':'Result reuse, saved-result remote replay, independent037 audit; no new model generation'},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(outputs))
