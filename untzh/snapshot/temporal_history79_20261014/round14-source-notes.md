# 第十四轮：实际参与者、报道日期和机构发言不能互相替代

继续固定随机40条，本地和untzh第十三轮状态均先读回；不换样、不训练、不重算神经模型分数。上一轮概念问答没有改变历史判定，本轮取得并保存新的来源证据。

## S35：温家宝会见的是总统特别代表

实际读取[外交部5月25日会见正文](https://www.fmprc.gov.cn/zyxw/201005/t20100525_308073.shtml)，明确温家宝当天在北京会见克林顿和盖特纳，他们是奥巴马的特别代表。[5月26日书面吹风稿](https://www.mfa.gov.cn/web/zyxw/201005/t20100526_308101.shtml)同样明确对话发生于24–25日，奥巴马发来书面致辞，温家宝会见美方代表团成员。

原编码Story8255477、BBC Monitoring Asia Pacific、sentence1同时有5月26日ObamaConsultWen、ObamaMakeVisitChina及互反事实。来源提示参与者映射或报道日期问题，但没有链接到该BBC原文，不能宣布这就是误码原因，也没有排除当天独立电话磋商。gold1→29与最终第一名Netanyahu仍保持真假未知，第一名前后未变。

实际读取[白宫加州讲话记录](https://obamawhitehouse.archives.gov/the-press-office/remarks-president-economy-0)：5月26日10:07 PDT位于Fremont，对应北京时间5月27日01:07。单个时刻不构成5月26日北京整日行程排除，更不能排除远程磋商，不据此追加严格负例。

## S18：美国发言与安理会行动须区分

实际读取[Iran Watch保存的当日国务院发言实录节选](https://www.iranwatch.org/library/government/united-states/executive-branch/department-state/us-dos-press-briefing-mccormack-negotiations-4-20-06)。发言者谈到制裁可能性、未来第七章决议及4月28日报告之后的安理会讨论；同时明确不能排除期间日常议事涉及伊朗。因此不能把美国发言直接赋给安理会，也不能凭预期未来会议反证当天所有威胁。原LeMondeStory34088749正文尚未连上，gold与最终Obama均未知。

联合国A/60/834、S/2006/273搜索预览不作正文证据：网页工具实际打开403，远程普通请求返回HTTP202、text/html、0字节，未取得PDF。GlobalSecurity实录镜像网页打开失败；远程请求结果另存，不拿摘要给候选赋标签。

远程两请求已自然结束：联合国空正文请求3.781秒；GlobalSecurity自然返回Network is unreachable，131.701秒。SSH会话73065自然exit0，回执round14-remote-fetch-receipts.json保存，没有待poll、重启或硬超时。

## 对比例的影响

本轮没有增加完整排名合理或错误的判定。原gold独立同日语义支持9条，最终rawTop1支持2条、严格证伪0条、未定38条。条件于gold和组织身份，完整错误漂移仍仅S38得到正下界；固定40样本错误下限2.5%不是总体点估计，合理变化占比仍未识别。不能把基准疑似误码或真实Top1直接当整张排名合理。

下一需要对应原新闻正文或实际filtered竞争者的同日真假，特别是低原rank且gold已支持的S06、S11、S36。现存评分只读复用，不以TRAIN首次出现日期作现实成立日期。
