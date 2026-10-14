# 第十六轮：已识别事件日期不能直接替代查询日期

固定随机40样本保持不变。S13/S15新增实际新闻正文，详细核对覆盖变为38条，1条详细来源获取失败，1条原编码身份核对。真假判定数没有增加：原gold有独立同日语义支持9条；最终rawTop1支持2条、明确证伪0条、未定38条；整张排序错误漂移仍仅S38获条件确认。总体合理/错误点估计仍为空，不能把来源覆盖增加写成错误率下降。

S13查询是“美国特使于2009-12-12与哪个国家磋商”，gold中国rank7→9，rawTop1苏丹前后未变。新读到[凤凰卫视原节目实录](https://phtv.ifeng.com/program/zbjsj/200912/1212_6349_1472239.shtml)：网页12月12日发布，节目为12月11日，所述博斯沃思北京会见发生在周五（12月11日）。这是已识别会面日期，不能直接支持12月12日gold，也没有排除另一次会面或另一名美国特使。[AP经CBS的正文](https://www.cbsnews.com/news/us-urges-patience-in-n-korea-talks/)给出北京活动，但相关会面未明确日期。原BBCMonitoring Story11283975句4同时编码北京磋商和平壤访问，提示回述背景；它不是原新闻正文，尚未证明报道身份。此前Gration行程不是对通用特使身份的确认。国务院目录仅搜索预览可见、正文获取失败，不以预览补真值。

S14查询是“IFRC于2014-10-25访问哪个国家”，gold中国rank1→2，伊朗是唯一新增filtered越位候选。新读到[IRNA经OANA的正文](https://oananews.org/content/news/general/veep-iran-ready-cooperation-humanitarian-fields)：12月18日报道的周三晚会面发生于12月17日。这支持一个真实的伊朗事件，但不能支持10月25日预测。[IFRC官方2014年报](https://www.ifrc.org/sites/default/files/Annual-report-2014.pdf)第12页的12月条目也记录伊朗合作备忘录；年度重点列表不是完整行程，不能据未列出而否认10月25日另一名代表访问。

S15查询是“谁于2014-08-27就伊朗发表声明”，gold伊朗媒体人员rank15→30，rawTop1 Obama前后未变。新实际读到[IranWatch保存的Psaki实录节选](https://www.iranwatch.org/library/governments/united-states/executive-branch/department-state/daily-press-briefing-spokesperson-jen-psaki-excerpts-11)，发言人是Psaki，不能当Obama本人当日声明。另读到[Reuters经BusinessRecorder的同日正文](https://www.brecorder.com/news/190911/iran-says-testing-new-nuclear-enrichment-machine-may-irk-west)，明确Salehi及Fars引用其离心机讲话；这提供当日伊朗媒体报道背景，但原BBCMonitoringStory35266399句3中的媒体人员身份与CAMEO行为主体没有对应。新闻转述者、被引述者、政府代表不能自动互换。[IranWatch的Fars摘要](https://www.iranwatch.org/news-brief/aeoi-chief-iran-ready-launch-new-generation-nuclear-centrifuges)仅为二级摘要，不冒称已读Fars原文。PressTV/Tasnim原页面获取失败，不由预览补真值。

三条的gold、最终答案和whole-rank判定均保持未知。已识别事件的其他日期或其他发言主体，属于不能直接用于本查询的证据；它不等于已证明整个查询假。网页回执、事件日期解释分别保存为round16-web-receipts.json、round16-web-receipts-tail.json、identified-event-dates-round16.json。

本轮未训练、未神经重评分、未改基准过滤，不执行独立审计、硬超时、SHA或归档。还尝试查询S21的精确2月4日商业支持来源，但没有取得可对应该日的新正文，不从不相关搜索结果赋标签。此前S38样本下界与总体条件置信下界保留原值。
