# 第十三轮：小幅排名变化的候选与来源日期

继续原固定随机40条；本地/远程权威状态已读回。没有换样、新训练、神经评分或重复模型评分。

## S39：新取得检察日报转载，未确认4月19日gold

[青岛新闻网转载检察日报正文](https://qingdaonews.com/content/2009-04/20/content_8033364.htm)于2009-04-20发布，正文明确曹建明18日在萨那讲话、与也门总检察长会谈、签署备忘录及会见其他官员。加强已有外交部报道的4月18日日期证据，没有获得4月19日另一次中也磋商或原内部Story10806432的全文身份链接。因此原gold仍为精确日期未知；伊拉克同日支持与条件补充过滤解释保留，不能将整个query升级为已证合理变化。

## S27：之前打不开的慰问正文取得，但官方签发在8月31日

通过untzh普通HTTPS实际取得[TheHabari慰问稿](https://www.thehabari.com/ikulu-yamtumia-rambirambi-rc-mbeya-vifo-vya-watu-10/)HTTP200全文；网页工具失败回执保留。9月1日转载正文含Kikwete对8月29日车祸的慰问与道路安全言论，本身没有明确新发言日期。

随后实际读到[总统府原始稿件](https://www.ikulu.go.tz/news/press-release-7)：文末为总统新闻部门签发“31 Agosti,2014”（2014年8月31日），顶部却显示“31st Jul,2014”。7月页面日期甚至早于正文8月29日事故。三个日期分别保留，不能用9月1日转载日期当新发言发生日期，也不能不加解释地使用官方页面顶部日期。

读取[总统府历史新闻档案](https://www.ikulu.go.tz/documents/press-releases?page=49)正文还确认：Mpwapwa的Gulwe桥奠基为8月29日，前一项Mbande/Kongwa活动为8月28日；9月2日公告称9月1日夜间到达内罗毕。后来翻页/滚动请求超时，但首次正文已经实际取得，记录两种结果。

S27原raw故事为AllAfrica Story35330934，地点Mpwapwa，总统statement在sentence3，Magufuli在sentence6。新读慰问稿不是已链接的原新闻，官方行程也未排除9月1日其他讲话。gold与两新增越位候选DomesticAffairs(Tanzania)、Ministry(Tanzania)仍未知。本条从来源获取失败改为详细正文核对，详细核对数36，获取失败3，身份上下文1；不增加gold或最终Top1真值标签。

Mpwapwa博客9月档案的普通远程HTTPS尝试自然返回Network is unreachable，131.615秒；SSH会话86227自然exit0，错误保存round13-mpwapwa-fetch-receipt.json。没有取得正文，不据索引预览赋标签，没有重启。另一两来源获取会话77305也自然exit0。没有待poll进程。

## S14及其他候选

ChinaDaily旧列表此次返回not-found页面，保留此前读取开头的回执，不将当前页面当历史全文。IFRC评论转载网页工具失败，远程普通HTTPS自然返回SSL UNEXPECTED_EOF；未取得正文。IFRC对中国/伊朗10月25日访问的真假仍未知。Abbas/Tsipras、Bortnikov/Fradkov和Angola/Portugal新检索未获得精确同日事实；其他日期旅行、签约或百科列表不赋真假标签。

本轮没有新增整张排序合理/错误判定。固定40错误逻辑下限仍为1/40=2.5%，总体下降查询中单侧95%条件下界仍55/42851=0.12835%；两总体点比例仍为空。尚未完成用户要求。
