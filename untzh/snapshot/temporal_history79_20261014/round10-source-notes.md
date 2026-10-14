# 第十轮：从错误成分收紧到错误排名漂移的下限

S38可以作更强的判断。既存评分缓存显示：更新前，未过滤排名中总共只有3个候选严格超过原答案，没有同分项；更新后，4个有组织成立时间反证的候选严格超过原答案。原答案“斯洛伐克外交部门于2009-09-17批评匈牙利外交部门”已有同期正文支持。

把所有现实正确的替代答案在两时期都过滤，剩下的排名只计算错误答案对gold的竞争。更新前这种错误竞争至多3个；更新后至少4个。因此现实正例补全后的排名下降至少1位。这比“存在4个错误成分”更强：即使其余未知竞争者都是正确答案，也无法消除全部错误漂移。

该推论不要求基准已过滤正例全部真实，因为更新前的上界来自未过滤评分。条件仍是：接受gold的同日关系语义映射、组织名/成员身份按事件发生时解释、比较已保存模型的候选空间。具体查询、三个更新前候选和四个错误竞争者见false-rank-drift-bounds-round10.json。远程只读既存数组0.082691秒，无模型调用。

固定随机40条中至少1条有这种可证明的错误漂移，样本条件下限2.5%。这是样本下限，不是总体点估计，也不表示其余97.5%合理。S04的8个错误候选还缺同日gold真实性；S21只有1个错误候选、原来已有1个候选在前，尚不能同样证明错误竞争总数增加。

S11：实际读[葡萄牙官方6月20日公报PDF](https://files.dre.pt/1s/2006/06/117a00/43634365.pdf)和[政府2007年批准文件](https://www.historico.portugal.gov.pt/pt/o-governo/arquivo-historico/governos-constitucionais/gc17/comunicados-cm/cm-2007/20070222.aspx)。前者为2002年CPLP章程修订，议会2006-04-20批准；后者明确双边融资合作在2006-04-05签署。[司法合作官方公告](https://diariodarepublica.pt/dr/detalhe/aviso/582-2006-655568)给出1995签署、2006-04-05换文、2006-05-05生效。它们可支持既存合作状态，不能支持ICEWS事件语义下6月20日发生一项新合作，也不排除另有同日合作。Portugal这一唯一越位答案仍未知。

S07：实际读[Business Recorder转载dpa正文](https://www.brecorder.com/news/3700920/no-winners-as-thaksin-heads-out-from-cambodia-20091116987187)，署名版权为dpa，非先前搜索推测的AFP；11月16日刊发，主要回述周六离境、前周外交争执。China只是地区协调背景，不能支持Thailand同日批评China，也不能用未提及制造负例。TNA博客另一路获取失败，摘要未用于标签。

S36：实际读挪威政府条约[第一章](https://www.regjeringen.no/no/dokumenter/stprp-nr-3-2007-2008-/id488644/?ch=1)和[背景章](https://www.regjeringen.no/no/dokumenter/stprp-nr-3-2007-2008-/id488644/?ch=2)，明确边界条约2007-07-11签署、此前3月/4月谈判；不能把签约日当6月8日会谈。[VG正文](https://www.vg.no/nyheter/utenriks/i/JKBM8/inviterte-putin-til-norge)只明确6月9日见Putin，6月11日更新时间不是发生日。

同源下载接续成功，实际提取并目视阅读6月13日《New Light of Myanmar》第4页MNA/Xinhua转载正文。Stockholm6月11日稿明确斯托尔滕贝格周五在莫斯科会见弗拉德科夫后宣布边界共识；最近周五为6月8日，与既存NTB时间表相符。支持S36原gold的同日Consult语义，原Top1与gold相同也获支持；没有宣称对应ITAR-TASS内部StoryID。原gold独立同日支持由8增至9。Bortnikov/Gryzlov与Fradkov会谈仍未知，不能因gold正确就判整个下降错误。

网页工具拒绝10.64MB PDF、远程curl自然TLS失败都保留为获取记录；本机下载实际522.091668秒，PyMuPDF提取及相关页渲染0.137108秒，无固定硬超时，原会话37917自然exit0。PDF/相关页文本/目视PNG/费用元数据均已保存。

总体合理/错误比例仍未识别。本轮固定40不换样、未训练、未重算神经评分，不把来源可得性子集的比例当总体估计。
