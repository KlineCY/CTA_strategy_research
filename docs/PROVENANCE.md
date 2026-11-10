# 数据与费用依据

## 研究数据

本项目使用37只ETF日线行情、资产类别清单、VIX与MOVE日线指数，以及有效联邦基金利率（EFFR）序列。ETF用于代表六类资产板块，回测收益是ETF价格收益。研究输入来自项目本地数据集。该数据集未标注37只ETF行情及两项波动率指数的上游供应商；仓库呈现数据结构、处理代码和汇总结果，原始CSV及EFFR JSON在本地保存。读取格式见[本地研究输入](../data/INPUTS.md)。

研究结果表记录已完成运行的汇总值。37只ETF覆盖1630个市场交易日，其中12日存在部分资产缺价。拆分调整、日期对齐、收益标签和缺失处理在运行时完成。现金基准所用EFFR文件的SHA256为`8df2e9a9f6a08fa5122f22e5632aaa0c3d2d0065ec28cb605c19d14def21ae22`；现金基准对该输入版本敏感。每次完整运行的输入、源码、配置、Python与依赖版本由`results/run_manifest.json`记录指纹。

## 成交费用

美国ETF佣金采用[Interactive Brokers的IBKR Pro Fixed美国股票与ETF费率](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php)作为研究代理：每股0.005美元、每笔最低1美元、成交额1％封顶。同日同ETF净成交量视作一笔订单。[盈透另一官方费率页面](https://www.interactivebrokers.com/en/accounts/fees/stknorthambundcoms.php)列示0.5％的上限；两页口径并存，研究账户固定采用前一页面的1％上限，以保证结果可以复算。该佣金模型不等于历史逐笔券商账单。

卖出端Section 31费率按照[美国证券交易委员会（SEC）的逐期公告](https://www.sec.gov/rules-regulations/fee-rate-advisories)计入。代码以交易日作为公告中收费生效日的研究代理，费率按每百万美元卖出成交额表示：

|生效日期|美元／百万美元|公告|
|---|---:|---|
|2020年1月1日|20.70|[2020财年公告](https://www.sec.gov/newsroom/press-releases/2020-7)|
|2020年2月18日|22.10|[2020财年公告](https://www.sec.gov/newsroom/press-releases/2020-7)|
|2021年2月25日|5.10|[2021财年调整令](https://www.sec.gov/file/34-90932)|
|2022年5月14日|22.90|[2022财年公告](https://www.sec.gov/newsroom/press-releases/2022-60)|
|2023年2月27日|8.00|[2023财年公告](https://www.sec.gov/newsroom/press-releases/2023-15)|
|2024年5月22日|27.80|[2024财年公告](https://www.sec.gov/rules-regulations/fee-rate-advisories/2024-2)|
|2025年5月14日|0.00|[2025财年公告](https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2)|
|2026年4月4日|20.60|[2026财年公告](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2)|

费用核算限于上述佣金和Section 31研究代理；ETF分红、融资、借券费用和滑点没有对应的逐日输入，未进入现金账本。
