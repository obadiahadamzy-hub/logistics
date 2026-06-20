# 物流企业AI智能运营决策平台

当前版本聚焦模块一：**可定制数据清洗流程中心**。本模块只属于学生1：数据工程师，用于完成数据导入、数据质量检查、数据清洗、配送时长处理和基础经营指标生成。

平台支持数据源扫描、质量检查、流程配置、批量清洗和多表连接，帮助企业将分散的原始数据加工为标准化数据资产。

## 安装依赖

```powershell
cd C:\Users\Administrator\Documents\物流人工智能技能大赛\logistics_ai_platform
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 生成模拟数据

```powershell
python scripts\generate_mock_data.py
```

生成到 `data/raw/` 的文件：

- `orders.csv`：7926行，含12条重复订单、6条数量缺失、8条日期格式不统一。
- `delivery.csv`：7926行，用于套用通用清洗规则并计算配送时长。
- `inventory.csv`：200行。
- `products.csv`：200行。
- `outlets.csv`：60行。
- `returns.csv`：537行。

## 启动系统

```powershell
streamlit run app.py
```

浏览器打开后进入：

```text
http://localhost:8501
```

## 模块一页面

1. **模块首页**  
   说明模块定位、学生岗位和演示路径。本模块不做ABC分类。

2. **数据源管理**  
   扫描 `data/raw/` 和 `data/processed/` 下的 CSV，显示行数、列数、缺失值、重复行，支持预览和上传 CSV。

3. **数据质量概览**  
   选择数据表后生成 `outputs/reports/data_quality_report.csv`，按字段汇总空值、重复和格式类问题。系统启动时会清理旧的质量报告，保证报告结果来自当前检查。

4. **数据预处理工作流**  
   先选择要处理的数据表，再在核心代码框中编写 `process(df)` 函数。运行后可查看处理后数据预览，并保存为可复用处理流程。保存流程后，可多选多张数据表完成批量清洗，生成对应的 `clean_*.csv`。

5. **数据连接模块**  
   以 `product_id` 为主键，把商品信息、销售指标、库存指标、退货指标合成一张商品基础指标表：

   - `products.csv`：商品主表，一行一个商品。
   - `orders.csv`：按 `product_id` 汇总销量、销售额、出库频次。
   - `inventory.csv`：按 `product_id` 汇总库存数量、库存金额。
   - `returns.csv`：按 `product_id` 汇总退货次数、退货率。
   - 输出：`data/cleaned/product_base_metrics.csv`。

   `product_base_metrics.csv` 不包含 `abc_class` 字段，ABC分类由模块二完成。

## 推荐演示顺序

1. 运行 `python scripts\generate_mock_data.py`。
2. 打开“数据源管理”，检查6张原始表。
3. 打开“数据质量概览”，生成质量报告。
4. 打开“数据预处理工作流”，保存处理流程，并多选数据表完成批量清洗。
5. 打开“数据连接模块”，生成 `product_base_metrics.csv`。

## 模块二单独启动

模块二入口文件为 `app2.py`，用于学生2“AI库存分析工程师”独立演示：

```bash
streamlit run app2.py
```

模块二会读取 `data/processed/product_base_metrics.csv`，如果该文件不存在，会自动兼容读取 `data/cleaned/product_base_metrics.csv`。新版流程先接收数据并等待构建数据大屏，再进入“大模型ABC智能分类”对话页生成Python代码，随后在“Python云平台”中粘贴/编辑代码并运行，生成 ABC分类结果、历史销量训练样本、销量预测模型训练评估流程、未来销量预测、库存预警、补货建议、清仓建议和高优先级商品交接文件。

## 模块三单独启动

模块三入口文件为 `app3.py`，用于学生3“配送智能规划工程师”独立演示：

```bash
streamlit run app3.py
```

模块三接收 `clean_delivery.csv`、`abnormal_delivery.csv`、`outlet_metrics.csv` 和 `high_priority_products.csv`。如果部分中间文件不存在，系统会基于现有订单、配送、网点和商品数据生成演示用结果。

模块三新版页面包含 7 个操作环节：数据接收与问题判断、异常订单诊断规则配置、网点异常汇总与风险评分、配送风险地图图层分析、K-Means聚类参数配置与评估、起降点候选规则配置、AI低空应急配送航线智能规划系统。异常订单诊断页会先加载 `product_base_metrics.csv` 辅助识别重点商品和生鲜冷链风险，并输出 `delivery_exception_result.csv` 和 `outlet_exception_summary.csv`；网点风险评分页面再读取异常汇总结果并合并 `outlet_metrics.csv` 生成 `outlet_risk_score.csv`。系统还会输出风险地图、聚类分区、候选起降点和低空应急航线规划结果。

## 模块四单独启动

模块四入口文件为 `app4.py`，用于学生4“大模型经营决策工程师”独立演示：

```bash
streamlit run app4.py
```

模块四接收前三个模块结果，提供 7 个页面：多模块结果接收与校验、经营问题选择与智能问答、大模型报告提示词编写、大模型模拟分析过程、智能经营诊断报告生成、决策建议驾驶舱、报告导出与比赛总结。第一页包含 RAG 经营知识库接入，可选择企业制度文档和模块结果文件，配置 Chunk 大小、重叠长度、Top-K、相似度阈值和检索方式，并生成 `rag_knowledge_base.csv` 与 `rag_retrieval_preview.csv`。涉及大模型问答的页面统一使用 `st.chat_input` 聊天输入框发起对话。第一版采用离线模板与预设回答，不需要联网或真实调用大模型；如果部分结果文件缺失，系统会自动补充演示数据，保证现场流程可跑通。
