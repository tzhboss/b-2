# 第九轮：实际发生日期、报道语义与错误排名必须分开

固定40条保持不变，使用已保存的5549个越过gold候选，不重训或神经重评分。新增检索回执完整保存在representative-search-receipts-round9.json，包括不相关结果和获取失败。

S32的基准目标是韩国在2009-10-13接待奥巴马，rank2→6。[美国国务院完整总统出访表](https://history.state.gov/departmenthistory/travels/president/obama-barack)把2009年韩国访问列为11月18—19日，10月2日丹麦访问后直到11月13日日本之行无其他出访条目。[白宫Zapatero会谈实录](https://obamawhitehouse.archives.gov/the-press-office/remarks-president-obama-and-president-zapatero-spain-after-meeting/)记10月13日13:21EDT椭圆办公室，[Fiesta Latina实录](https://obamawhitehouse.archives.gov/realitycheck/the-press-office/remarks-president-performance-white-house-fiesta-latina)记10月13日19:30EDT南草坪（发布头为14日）。两时刻在韩国均已是14日，不能单凭它们声称排除13日KST每一时刻；负向判断依赖官方出访表的覆盖及实际访问解释。[CAMEO042/043说明](https://eventdata.parusanalytics.com/cameo.dir/04consult.html)给出旅行/接待语义。

在“事件日表示实际发生日、043表示实际到访、官方总统出访记录覆盖实际访问”的条件下，这条原gold有明确日期不相容。原AFP Story10206966正文仍未取得，是否是提前报道11月访问、自动编码如何处理将来时尚未确定。因此保存新的gold_literal_occurrence_truth及evaluation_target_counterindication，原gold_truth在原报道/编码语义上仍未知。固定40中1条具有此条件日期反证（2.5%是覆盖数量），不称总体数据错误率，更不将gold下降自动当合理调整。最终Karimov仍未知。

S32实际四个越位是Karimov/MackySall/China/Xi，均原候选。Xi分差0.1214561462。[外交部正文](https://www.mfa.gov.cn/web/wjdt_674879/gjldrhd_674881/200910/t20091014_7699752.shtml)明确他10月13日在德国出席法兰克福书展开幕、会见出版界并参观歌德故居，另有12日德国会谈、14日抵保行程。这是候选行程反向证据，但未取得覆盖查询日期全部时刻的独立排除证据；不新增严格错误竞争者标签。

S34唯一越位是阮春福，gold越南接待老挝rank1→2。新读[SGGP同期越南语正文](https://www.sggp.org.vn/tphcm-mit-tinh-trong-the-ky-niem-30-nam-giai-phong-mien-nam-thong-nhat-dat-nuoc-post298206.html)，明确4月30日上午胡志明市庆典，老挝由国防部长Douangchay Phichit率团，另有柬埔寨代表；5月1日刊发，与raw同句胡志明市、老挝/柬埔寨语境相似。不能因此认定5月3日有新访问或排除阮春福参与；原Thai News Service Story6708924仍未连接，保持未知。

远程两实体原身份扫描5.805915秒自然exit0：National Salvation Government的raw country为Libya，但6月7日及6月25/26日已有命名，后两条主体来自伊拉克。[2014年第46号组阁决定正文](https://security-legislation.ly/latest-laws/resolution-no-46-of-2014-on-forming-a-national-salvation-government/)记9月2日，另读[美国国务院2014利比亚报告正文转载](https://www.ecoi.net/en/document/1058946.html)记8月25日宣布Omar al-Hassi政府。早于正式成立的raw可能是拟议政府/泛称/自动身份混淆，不将其作为S38第五个严格见证。UN原网页403，摘要不赋标签。Democratic Alliance (Africa)的raw country为空、报道语境多为南非，也不能套另一个晚成立政党制造错误。新增两项明确排除条件，累计严格时间见证仍13，S38仍4位置（162→158）。

S21替代议会PDF地址仍403/受限；检索摘要中的历年支持不能当2月4日新意向。保留失败记录，不修改标签。首远程状态读取的python -c参数被SSH shell拆开而语法失败，未运行读取或改远程文件；改用PowerShell literal here-string送python stdin后读回成功，未训练重试。

目前34条详细正文核对、5条获取失败、1条编码身份核对。原gold同日支持8、最终rawTop1支持2/明确证伪0/未知38；world合理率/真正模型错误率仍null。1条错误排名成分、1条漏标消除漂移、1条原目标实际日期反证属于不同证据轴，不能拼成一个三分比例。目标继续active。
