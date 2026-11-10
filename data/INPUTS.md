# 本地研究输入

本仓库不分发ETF行情、VIX／MOVE指数或EFFR原始文件。运行者须在`data/raw/`提供具有使用权的41个输入文件；输入文件名和字段须与读取程序一致。所附汇总CSV是既有研究结果，不构成原始行情的替代数据。

|输入|数量|文件名|必要字段|
|---|---:|---|---|
|ETF日线|37|`{symbol}_ohlcv_1d.csv`|`ts_event,open,high,low,close`|
|资产清单|1|`_manifest.csv`|`symbol,category`|
|VIX日线|1|`VIX_index_1d_2020_now.csv`|`ts_event`或`date`，以及`close`|
|MOVE日线|1|`MOVE_index_1d_2020_now.csv`|`ts_event`或`date`，以及`close`|
|EFFR利率|1|`EFFR_2020_2026.json`|`refRates`数组内的`effectiveDate,percentRate`|

ETF文件按代码中的`*_ohlcv_1d.csv`模式识别，预期恰好37个。时间戳解析为UTC；重复日期、非正OHLC价格和缺失资产类别均会触发输入校验。ETF覆盖区间与资产构成若不同于历史研究输入，重新运行的参数选择和绩效可能变化。EFFR读取程序登记了原研究文件的SHA256：`8df2e9a9f6a08fa5122f22e5632aaa0c3d2d0065ec28cb605c19d14def21ae22`。该指纹用于保证现金基准与历史结果一致；不同版本的EFFR文件不能直接用于现有账户复算。
