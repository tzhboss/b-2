# 第八轮：逐候选的真假证据与排名影响

本轮沿用40条固定样本及5549个既存新越位候选，未训练或重新评分。针对Tasnim news agency、Militant (Ahrar ash-Sham)、Jaysh al-Islam和Social Democratic Party (China)，远程只读其TRAIN首次年份原始记录，6.774842秒；不是全库新匹配或总体运行费用。来源全文检索、解释与保存费用另计。

## 新增六个组织时间反证

[Tasnim自己发布的周年文章](https://www.tasnimnews.ir/fa/news/1398/08/22/2140097/کدخدایی-سالگرد-تأسیس-خبرگزاری-تسنیم-را-تبریک-گفت)正文明确成立于伊朗历1391年8月22日（2012年）；[英文About页面](https://www.tasnimnews.ir/en/about)说明2013年英文启动约在波斯语部门开启9个月之后。此处采用2012年的保守年度界限，不只靠英文网站上线日期。原始记录确认实体是伊朗媒体通讯社。

[Ahrar al-Sham研究档案](https://mappingmilitants.org/files/group-profiles/ahrar_al-sham.pdf)摘要和叙述明确组织2011年成立，叙述具体说2011年末；网页字段January1不是精确日证据。原始记录确认候选是叙利亚该组织的成员类别。

[Liwa al-Islam研究档案](https://mappingmilitants.org/mmp-group/liwa-al-islam)正文记前身2011年成立、2013年9月合并为Jaysh al-Islam。因此对2005/2009查询，即使允许前身别名解释，仍早于前身成立。原始记录确认候选是叙利亚叛乱组织。

三者分别在S04（2005-05-16）和S38（2009-09-17）新增超过gold，共六个新见证，加上既有七个为十三个时间不相容候选，涉及仍是S04/S21/S38三条；不能把同一查询的多个候选当更多独立样本。组织/当时成员语义条件保持，不能把后来成员身份回溯当时所属组织。

S38原gold为斯洛伐克外交部当天批评匈牙利外交部，已有同日SITA/ČTK正文支持。新三候选gap分别为Tasnim3.297352076、Ahrar成员1.036069155、Jaysh0.133186579，均是当时候选空间不存在的新行。加原有Nusra成员类别，现有4个明确错误排名位置。只从score列表移除四者，after rank162→158，before仍4；净下降158个位置中已解释4个、其余154个尚未解释。4/158只是该查询的已确认错误位置贡献，不是总体错误率或全部错误成分比例。S04八个错误竞争者、S21一个，但其原gold同日真实性仍待确定。

## 排除错误的同名映射

Social Democratic Party (China)原始国家China，并有2010-10-04四条记录。不能把它直接认作2015年成立的台湾社会民主党。成立时间见证排除，保留身份未知。

## S17的部分漏标解释仍有日期和主体条件

[Korea Times正文](https://www.koreatimes.co.kr/southkorea/20101110/korea-russia-agree-to-boost-energy-ties)标2010-11-10 19:05KST，结尾报告Lee将会见英/德/美/中领导人；可以支持在刊发日已宣布拟会谈的语义解释，德/美对应Merkel/Obama。但同句today与实际11日会谈存在日期措辞问题，也不是Lee本人直接引述，所以此处只记条件支持，不将其赋为严格同日真值。[AP转载](https://6abc.com/archive/7779554/)显示美国日期10日、正文Thursday活动；[CBS日程](https://www.cbsnews.com/news/president-obamas-schedule-in-seoul-11-10-11-10/)明确全用美国东部时间，不能把实际会谈统一认成首尔10日。

旧缓存确认Obama/Merkel分别是新越过Medvedev的合法候选，aftergap0.548259735/0.300675392，beforegap均负。在接受“刊发当天已宣布Lee拟会谈”的条件下，补充过滤它们使gold after rank8→6；原rank1，仍有5个新越位候选未知。这不是整条排名下降合理或新增两个严格真答案判定。最终rawTop1日本仍已被filtered排除、现实真假未定。

全40的最终Top1支持2/证伪0/未知38，gold同日支持8仍不变。真实合理/错误总体比例仍null。十三见证、三条查询覆盖率、单条四位置贡献、条件漏标情景为不同量，不能混为一张合理/错误饼图。产物temporal-competitor-evidence.json / partial-positive-competitor-round8.json / competitor-identity-context-round8.json / representative-search-receipts-round8.json。
