from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


APP_TITLE = "物流企业AI智能运营决策平台"
MODULE_TITLE = "大模型经营决策与智能报告生成系统"
PAGE_TITLE = "物流企业大模型经营问答与决策报告系统"
OPERATOR = "学生4：大模型经营决策工程师"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = DATA_DIR / "cleaned"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUTS_DIR / "reports"
LOG_DIR = OUTPUTS_DIR / "logs"
LOG_FILE = LOG_DIR / "decision_llm_log.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"
RAG_KB_FILE = REPORT_DIR / "rag_knowledge_base.csv"
RAG_RETRIEVAL_FILE = REPORT_DIR / "rag_retrieval_preview.csv"

PAGES = [
    "多模块结果接收与校验",
    "经营问题选择与智能问答",
    "大模型报告编写与生成",
    "AI智能决策驾驶舱",
]

DEFAULT_METRICS = {
    "product_total": 200,
    "a_products": 32,
    "b_products": 58,
    "c_products": 110,
    "a_sales_rate": 70.4,
    "b_sales_rate": 19.3,
    "c_sales_rate": 10.3,
    "c_stock_rate": 45.5,
    "shortage_products": 18,
    "normal_stock_products": 86,
    "overstock_products": 58,
    "high_overstock_products": 38,
    "purchase_amount": 88000,
    "release_amount": 126000,
    "delivery_records": 7914,
    "exception_orders": 312,
    "long_timeout_orders": 96,
    "a_timeout_orders": 46,
    "fresh_timeout_orders": 28,
    "township_timeout_orders": 72,
    "outlet_total": 60,
    "high_risk_outlets": 9,
    "medium_risk_outlets": 17,
    "low_risk_outlets": 34,
    "township_high_risk_outlets": 7,
    "cluster_count": 4,
    "launch_candidates": 8,
    "launch_points": 3,
    "route_planned_orders": 28,
    "route_covered_orders": 22,
    "route_uncovered_orders": 6,
    "avg_saved_minutes": 38,
}

DEFAULT_PROMPT = """请你作为物流企业经营决策顾问，基于数据质量报告、ABC分类结果、销量预测结果、库存预警结果、配送异常诊断结果、网点风险评分结果、K-Means聚类分区结果、低空起降点推荐结果和低空航线规划结果，生成一份面向企业管理层的综合经营诊断报告。报告需要说明核心问题、数据依据、库存优化建议、配送优化建议、低空应急配送试点建议和预期价值。"""

REPORT_TYPES = ["库存优化报告", "配送风险报告", "低空应急配送报告", "综合经营诊断报告"]

ANALYSIS_STEPS = [
    "大模型正在读取模块一数据质量报告...",
    "大模型正在提取商品基础指标...",
    "大模型正在检索RAG经营知识库...",
    "大模型正在分析ABC分类结构...",
    "大模型正在识别缺货与积压风险...",
    "大模型正在读取配送异常订单...",
    "大模型正在分析网点风险评分...",
    "大模型正在解析K-Means配送片区结果...",
    "大模型正在读取低空起降点与航线规划结果...",
    "大模型正在生成经营问题诊断...",
    "大模型正在生成决策建议...",
    "大模型正在组织报告结构...",
]

REPORT_OUTLINE = [
    "一、项目背景与数据来源",
    "二、企业当前经营核心问题",
    "三、库存结构与商品经营分析",
    "四、配送异常与网点风险分析",
    "五、低空应急配送规划分析",
    "六、综合决策建议",
    "七、预期价值与实施路径",
]

REQUIRED_FILES = [
    ("模块一 数据工程师", "data_quality_report.csv", "数据质量报告", "说明数据清洗和质量情况"),
    ("模块一 数据工程师", "clean_orders.csv", "清洗订单表", "支撑订单销售和经营分析"),
    ("模块一 数据工程师", "product_base_metrics.csv", "商品基础指标表", "支撑商品经营分析"),
    ("模块一 数据工程师", "outlet_metrics.csv", "网点基础指标表", "支撑网点基础情况分析"),
    ("模块二 AI库存分析工程师", "abc_summary.csv", "ABC分类汇总表", "说明商品结构"),
    ("模块二 AI库存分析工程师", "abc_result.csv", "ABC分类明细表", "查看具体商品分类"),
    ("模块二 AI库存分析工程师", "forecast_result.csv", "销量预测结果表", "支撑未来销量判断"),
    ("模块二 AI库存分析工程师", "inventory_warning.csv", "库存风险预警表", "说明缺货、积压风险"),
    ("模块二 AI库存分析工程师", "purchase_advice.csv", "补货建议表", "生成采购建议"),
    ("模块二 AI库存分析工程师", "clearance_advice.csv", "清仓建议表", "生成库存优化建议"),
    ("模块二 AI库存分析工程师", "high_priority_products.csv", "高优先级商品表", "说明重点保障商品"),
    ("模块三 配送智能规划工程师", "delivery_exception_result.csv", "配送异常订单诊断表", "说明异常配送订单"),
    ("模块三 配送智能规划工程师", "outlet_exception_summary.csv", "网点异常订单汇总表", "说明异常订单在网点上的分布"),
    ("模块三 配送智能规划工程师", "outlet_risk_score.csv", "网点风险评分表", "说明高风险网点"),
    ("模块三 配送智能规划工程师", "cluster_result.csv", "网点聚类结果表", "说明配送优化片区"),
    ("模块三 配送智能规划工程师", "launch_candidates.csv", "候选起降点表", "说明每个片区候选起降点"),
    ("模块三 配送智能规划工程师", "launch_points.csv", "起降点推荐表", "说明最终低空起降点"),
    ("模块三 配送智能规划工程师", "route_candidates.csv", "候选航线表", "说明多条候选航线"),
    ("模块三 配送智能规划工程师", "route_assignment.csv", "航线路径分配表", "说明重点订单航线分配"),
    ("模块三 配送智能规划工程师", "route_simulation_report.html", "低空航线仿真报告", "展示低空配送可视化结果"),
]


def ensure_dirs() -> None:
    for folder in [PROCESSED_DIR, CLEANED_DIR, REPORT_DIR, LOG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_log(action: str, input_files: str, output_file: str, status: str = "success", remark: str = "") -> None:
    ensure_dirs()
    row = pd.DataFrame(
        [
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "module": "模块四",
                "operator": OPERATOR,
                "action": action,
                "input_files": input_files,
                "output_file": output_file,
                "status": status,
                "remark": remark,
            }
        ]
    )
    if LOG_FILE.exists():
        old = read_csv(LOG_FILE)
        row = pd.concat([old, row], ignore_index=True)
    write_csv(row, LOG_FILE)


def scroll_to_page_bottom() -> None:
    components.html(
        """
        <script>
        const main = window.parent.document.querySelector('section.main');
        if (main) {
            setTimeout(() => main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' }), 80);
        } else {
            setTimeout(() => window.parent.scrollTo({ top: window.parent.document.body.scrollHeight, behavior: 'smooth' }), 80);
        }
        </script>
        """,
        height=0,
    )


def render_timed_steps(steps: list[str], duration_seconds: float, title: str = "执行过程") -> list[str]:
    progress = st.progress(0)
    log_box = st.empty()
    lines: list[str] = []
    sleep_seconds = duration_seconds / max(len(steps), 1)
    for idx, step in enumerate(steps, start=1):
        lines.append(step)
        log_box.code("\n".join(lines), language="text")
        progress.progress(idx / len(steps))
        scroll_to_page_bottom()
        time.sleep(sleep_seconds)
    return lines


def stream_markdown_output(markdown_text: str, duration_seconds: float = 10.0) -> None:
    st.caption("大模型正在逐步生成报告内容...")
    progress = st.progress(0)
    output_box = st.empty()
    steps = min(max(len(markdown_text) // 80, 36), 80)
    sleep_seconds = duration_seconds / max(steps, 1)
    for idx in range(1, steps + 1):
        cut = max(1, int(len(markdown_text) * idx / steps))
        output_box.markdown(markdown_text[:cut])
        progress.progress(idx / steps)
        if idx % 8 == 0 or idx == steps:
            scroll_to_page_bottom()
        time.sleep(sleep_seconds)


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def sample_doc_files() -> list[Path]:
    if not SAMPLE_DOCS_DIR.exists():
        return []
    return sorted(SAMPLE_DOCS_DIR.glob("*.txt"))


def candidate_paths(file_name: str) -> list[Path]:
    if file_name.endswith(".html"):
        return [REPORT_DIR / file_name, PROCESSED_DIR / file_name, CLEANED_DIR / file_name]
    if file_name == "data_quality_report.csv":
        return [REPORT_DIR / file_name, PROCESSED_DIR / file_name, CLEANED_DIR / file_name]
    return [PROCESSED_DIR / file_name, CLEANED_DIR / file_name, REPORT_DIR / file_name, RAW_DIR / file_name]


def find_file(file_name: str) -> Path | None:
    return next((path for path in candidate_paths(file_name) if path.exists()), None)


def preferred_output_path(file_name: str) -> Path:
    if file_name.endswith(".html"):
        return REPORT_DIR / file_name
    if file_name == "data_quality_report.csv":
        return REPORT_DIR / file_name
    return PROCESSED_DIR / file_name


def file_status() -> pd.DataFrame:
    rows = []
    for module, file_name, display_name, desc in REQUIRED_FILES:
        found = find_file(file_name)
        rows.append(
            {
                "module": module,
                "file_name": file_name,
                "display_name": display_name,
                "description": desc,
                "status": "已接收" if found else "待生成",
                "path": str(found.relative_to(PROJECT_ROOT)) if found else str(preferred_output_path(file_name).relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def render_file_status_table(status: pd.DataFrame, height: int = 220) -> None:
    rows = []
    for _, row in status.iterrows():
        badge_bg = "rgba(66, 198, 164, 0.14)" if row.get("status") == "已接收" else "rgba(148, 163, 184, 0.12)"
        badge_color = "#a7f3d0" if row.get("status") == "已接收" else "#a8b4c4"
        module_name = html.escape(str(row.get("module", "")))
        display_name = html.escape(str(row.get("display_name", row.get("file_name", ""))))
        status_text = html.escape(str(row.get("status", "")))
        rows.append(
            "<tr>"
            f"<td>{module_name}</td>"
            f"<td>{display_name}</td>"
            f"<td><span style='display:inline-block;padding:2px 8px;border-radius:999px;background:{badge_bg};color:{badge_color};font-size:12px;'>{status_text}</span></td>"
            "</tr>"
        )
    table_html = f"""
    <div style="max-height:{height}px; overflow-y:auto; border:1px solid #243244; border-radius:8px; background:#0f172a;">
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead style="position:sticky; top:0; background:#172235; z-index:1;">
          <tr>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #345069; color:#dce7f5;">来源</th>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #345069; color:#dce7f5;">数据文件</th>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #345069; color:#dce7f5;">状态</th>
          </tr>
        </thead>
        <tbody style="color:#d7e3f3;">
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def to_number(series: pd.Series, default: float = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = " ".join(text.replace("\r", "\n").split())
    if not cleaned:
        return []
    chunk_size = max(chunk_size, 80)
    overlap = min(max(overlap, 0), chunk_size // 2)
    chunks = []
    start = 0
    while start < len(cleaned):
        chunks.append(cleaned[start : start + chunk_size])
        if start + chunk_size >= len(cleaned):
            break
        start += chunk_size - overlap
    return chunks


def result_file_options() -> list[str]:
    preferred = [
        "data_quality_report.csv",
        "product_base_metrics.csv",
        "outlet_metrics.csv",
        "abc_summary.csv",
        "abc_result.csv",
        "forecast_result.csv",
        "inventory_warning.csv",
        "purchase_advice.csv",
        "clearance_advice.csv",
        "high_priority_products.csv",
        "delivery_exception_result.csv",
        "outlet_exception_summary.csv",
        "outlet_risk_score.csv",
        "cluster_result.csv",
        "launch_candidates.csv",
        "launch_points.csv",
        "route_candidates.csv",
        "route_assignment.csv",
    ]
    return [file_name for file_name in preferred if find_file(file_name) is not None]


def business_summary_entries(metrics: dict[str, float | int]) -> list[tuple[str, str, str]]:
    return [
        (
            "库存经营知识",
            "ABC分类与库存风险",
            f"A类商品{metrics['a_products']}种，贡献{metrics['a_sales_rate']}%销售额；C类商品{metrics['c_products']}种，销售贡献{metrics['c_sales_rate']}%，库存金额占比{metrics['c_stock_rate']}%。缺货风险商品{metrics['shortage_products']}种，建议清仓商品{metrics['high_overstock_products']}种。",
        ),
        (
            "配送风险知识",
            "配送异常诊断",
            f"异常配送订单{metrics['exception_orders']}条，其中A类重点商品超时{metrics['a_timeout_orders']}条，生鲜冷链超时{metrics['fresh_timeout_orders']}条，超长配送订单{metrics['long_timeout_orders']}条。",
        ),
        (
            "网点风险知识",
            "网点异常汇总与风险评分",
            f"高风险网点{metrics['high_risk_outlets']}个，其中乡镇高风险网点{metrics['township_high_risk_outlets']}个。风险来源包括道路等级、服务能力、历史超时率和重点商品超时。",
        ),
        (
            "低空规划知识",
            "低空应急配送",
            f"K-Means将网点划分为{metrics['cluster_count']}个配送优化片区，筛选候选起降点{metrics['launch_candidates']}个，推荐起降点{metrics['launch_points']}个；纳入低空规划订单{metrics['route_planned_orders']}单，可低空覆盖{metrics['route_covered_orders']}单。",
        ),
        (
            "经营决策知识",
            "综合行动建议",
            "企业应优先补货A类缺货商品、清仓C类高库存商品、优先处理A类和生鲜冷链超时订单，并在乡镇高风险片区试点低空应急配送。",
        ),
    ]


def build_rag_knowledge_base(
    doc_names: list[str],
    result_files: list[str],
    chunk_size: int,
    overlap: int,
    metrics: dict[str, float | int],
) -> pd.DataFrame:
    rows = []
    knowledge_index = 1

    for doc_path in sample_doc_files():
        if doc_names and doc_path.name not in doc_names:
            continue
        for chunk_index, chunk in enumerate(chunk_text(read_text_file(doc_path), chunk_size, overlap), start=1):
            rows.append(
                {
                    "knowledge_id": f"K{knowledge_index:04d}",
                    "source_type": "企业制度文档",
                    "source_file": doc_path.name,
                    "knowledge_category": doc_path.stem,
                    "chunk_index": chunk_index,
                    "content": chunk,
                    "keyword_tags": "制度,流程,规范,低空,配送,库存,退货",
                }
            )
            knowledge_index += 1

    for category, source_file, content in business_summary_entries(metrics):
        rows.append(
            {
                "knowledge_id": f"K{knowledge_index:04d}",
                "source_type": "多模块指标摘要",
                "source_file": source_file,
                "knowledge_category": category,
                "chunk_index": 1,
                "content": content,
                "keyword_tags": "库存,配送,低空,网点,经营决策",
            }
        )
        knowledge_index += 1

    for file_name in result_files:
        path = find_file(file_name)
        if path is None or file_name.endswith(".html"):
            continue
        try:
            df = read_csv(path)
        except Exception:
            continue
        summary = f"{file_name} 包含{len(df)}行、{len(df.columns)}列，主要字段包括：{', '.join(df.columns[:12])}。"
        if file_name == "abc_summary.csv" and {"abc_class", "product_count"}.issubset(df.columns):
            summary += " ABC分类汇总用于判断A类重点商品、B类稳定商品和C类库存占用商品。"
        if file_name == "outlet_risk_score.csv" and "risk_level" in df.columns:
            high_count = int((df["risk_level"].astype(str) == "高风险").sum())
            summary += f" 其中高风险网点{high_count}个，用于配送资源和低空试点决策。"
        if file_name == "route_assignment.csv":
            summary += " 航线路径分配表用于判断哪些高优先级异常订单可纳入低空应急配送。"
        rows.append(
            {
                "knowledge_id": f"K{knowledge_index:04d}",
                "source_type": "结果数据表",
                "source_file": file_name,
                "knowledge_category": "数据结果摘要",
                "chunk_index": 1,
                "content": summary,
                "keyword_tags": "数据表,指标,报告,RAG",
            }
        )
        knowledge_index += 1

    knowledge = pd.DataFrame(
        rows,
        columns=["knowledge_id", "source_type", "source_file", "knowledge_category", "chunk_index", "content", "keyword_tags"],
    )
    write_csv(knowledge, RAG_KB_FILE)
    write_log("构建RAG经营知识库", ";".join(doc_names + result_files), "rag_knowledge_base.csv", remark=f"知识条目{len(knowledge)}条")
    return knowledge


def retrieve_rag_context(query: str, top_k: int, threshold: float) -> pd.DataFrame:
    if not RAG_KB_FILE.exists():
        return pd.DataFrame(columns=["knowledge_id", "source_file", "knowledge_category", "similarity_score", "content"])
    knowledge = read_csv(RAG_KB_FILE)
    if knowledge.empty:
        return knowledge

    keywords = ["库存", "补货", "清仓", "配送", "超时", "低空", "起降点", "A类", "C类", "冷链", "网点", "利润", "销量", "退货", "风险", "报告"]
    active_keywords = [word for word in keywords if word in query] or ["经营", "风险", "决策"]

    scored = knowledge.copy()
    scores = []
    for _, row in scored.iterrows():
        text = f"{row.get('content', '')} {row.get('keyword_tags', '')} {row.get('knowledge_category', '')}"
        hit_count = sum(1 for word in active_keywords if word in text)
        score = min(0.99, 0.55 + hit_count * 0.12)
        scores.append(score)
    scored["similarity_score"] = scores
    retrieved = scored[scored["similarity_score"] >= threshold].sort_values("similarity_score", ascending=False).head(top_k)
    if retrieved.empty:
        retrieved = scored.sort_values("similarity_score", ascending=False).head(top_k)
    output = retrieved[["knowledge_id", "source_file", "knowledge_category", "similarity_score", "content"]].copy()
    write_csv(output, RAG_RETRIEVAL_FILE)
    write_log("执行RAG知识召回", query, "rag_retrieval_preview.csv", remark=f"TopK={top_k}")
    return output


def ensure_demo_outputs() -> None:
    ensure_dirs()
    m = DEFAULT_METRICS

    if find_file("data_quality_report.csv") is None:
        write_csv(
            pd.DataFrame(
                [
                    ("重复订单", "orders.csv", 12, "按 order_id 去重"),
                    ("字段为空", "orders.csv", 6, "补全 quantity 字段"),
                    ("格式不统一", "orders.csv", 8, "统一日期格式"),
                ],
                columns=["issue_type", "dataset_name", "issue_count", "suggestion"],
            ),
            REPORT_DIR / "data_quality_report.csv",
        )

    if find_file("clean_orders.csv") is None:
        rows = []
        for idx in range(1, 101):
            rows.append(
                {
                    "order_id": f"ORD{idx:06d}",
                    "order_date": "2026-06-15",
                    "customer_id": f"C{idx % 60 + 1:04d}",
                    "product_id": f"P{idx % 200 + 1:04d}",
                    "quantity": 1 + idx % 5,
                    "order_amount": round((1 + idx % 5) * 18.8, 2),
                    "outlet_id": f"O{idx % 60 + 1:03d}",
                    "order_status": "已签收",
                    "is_urgent": 1 if idx % 12 == 0 else 0,
                    "clean_status": "演示清洗通过",
                }
            )
        write_csv(pd.DataFrame(rows), PROCESSED_DIR / "clean_orders.csv")

    if find_file("product_base_metrics.csv") is None:
        rows = []
        classes = ["A"] * m["a_products"] + ["B"] * m["b_products"] + ["C"] * m["c_products"]
        categories = ["生鲜粮油", "冷链食品", "农产品", "零食饮料", "日用品", "厨具", "小家电配件"]
        for idx, cls in enumerate(classes, start=1):
            sales = 1200 - idx * 2 if cls == "A" else 520 - idx if cls == "B" else 80 + idx % 30
            rows.append(
                {
                    "product_id": f"P{idx:04d}",
                    "product_name": f"演示商品{idx:03d}",
                    "category": categories[idx % len(categories)],
                    "sales_30d": max(int(sales * 0.35), 1),
                    "sales_90d": max(int(sales), 1),
                    "sales_amount_90d": max(float(sales * 18), 10),
                    "out_freq_90d": max(int(sales / 4), 1),
                    "stock_qty": 80 + idx % 180,
                    "stock_amount": float((80 + idx % 180) * 12),
                    "gross_profit_rate": 0.18,
                    "return_count": idx % 5,
                    "return_rate": round((idx % 5) / 100, 4),
                    "stock_turnover_days": 20 if cls == "A" else 65 if cls == "B" else 150,
                }
            )
        write_csv(pd.DataFrame(rows), PROCESSED_DIR / "product_base_metrics.csv")

    if find_file("abc_summary.csv") is None:
        write_csv(
            pd.DataFrame(
                [
                    ("A", 32, 930009.89, 0.704, 960356.32, 0.286, "重点保障商品，需优先补货"),
                    ("B", 58, 254960.10, 0.193, 869693.31, 0.259, "稳定销售商品，维持正常库存"),
                    ("C", 110, 136066.79, 0.103, 1527839.60, 0.455, "低贡献商品，存在库存占用风险"),
                ],
                columns=["abc_class", "product_count", "sales_amount", "sales_amount_rate", "stock_amount", "stock_amount_rate", "management_strategy"],
            ),
            PROCESSED_DIR / "abc_summary.csv",
        )

    if find_file("abc_result.csv") is None:
        base = read_csv(find_file("product_base_metrics.csv") or PROCESSED_DIR / "product_base_metrics.csv")
        classes = ["A"] * m["a_products"] + ["B"] * m["b_products"] + ["C"] * m["c_products"]
        result = base.head(len(classes)).copy()
        result["cum_rate"] = [min((idx + 1) / len(classes), 1) for idx in range(len(result))]
        result["abc_class"] = classes[: len(result)]
        write_csv(result, PROCESSED_DIR / "abc_result.csv")

    if find_file("inventory_warning.csv") is None:
        abc = read_csv(find_file("abc_result.csv") or PROCESSED_DIR / "abc_result.csv")
        risk_list = ["缺货风险"] * 18 + ["库存正常"] * 86 + ["库存积压"] * 58 + ["高库存积压"] * 38
        warning = abc.head(len(risk_list)).copy()
        warning["inventory_risk"] = risk_list[: len(warning)]
        warning["forecast_30d"] = warning.get("sales_30d", pd.Series([30] * len(warning)))
        warning["priority_level"] = warning["inventory_risk"].map({"缺货风险": "高", "高库存积压": "中"}).fillna("低")
        write_csv(warning, PROCESSED_DIR / "inventory_warning.csv")

    if find_file("forecast_result.csv") is None:
        warning = read_csv(find_file("inventory_warning.csv") or PROCESSED_DIR / "inventory_warning.csv")
        forecast = warning.copy()
        forecast["model_name"] = "XGBoost销量预测模型"
        forecast["forecast_7d"] = (to_number(forecast["forecast_30d"]) / 30 * 7).round(1)
        write_csv(forecast, PROCESSED_DIR / "forecast_result.csv")

    if find_file("purchase_advice.csv") is None:
        warning = read_csv(find_file("inventory_warning.csv") or PROCESSED_DIR / "inventory_warning.csv")
        purchase = warning[warning["inventory_risk"] == "缺货风险"].head(18).copy()
        purchase["suggest_purchase_qty"] = 120
        purchase["suggest_purchase_amount"] = round(DEFAULT_METRICS["purchase_amount"] / max(len(purchase), 1), 2)
        write_csv(purchase, PROCESSED_DIR / "purchase_advice.csv")

    if find_file("clearance_advice.csv") is None:
        warning = read_csv(find_file("inventory_warning.csv") or PROCESSED_DIR / "inventory_warning.csv")
        clearance = warning[warning["inventory_risk"] == "高库存积压"].head(38).copy()
        clearance["clearance_strategy"] = "降价促销、组合销售、停止采购"
        clearance["expected_release_amount"] = round(DEFAULT_METRICS["release_amount"] / max(len(clearance), 1), 2)
        write_csv(clearance, PROCESSED_DIR / "clearance_advice.csv")

    if find_file("high_priority_products.csv") is None:
        abc = read_csv(find_file("abc_result.csv") or PROCESSED_DIR / "abc_result.csv")
        high = abc[abc.get("abc_class", pd.Series(dtype=str)).astype(str).eq("A")].head(28).copy()
        high["priority_reason"] = "A类重点商品或存在缺货风险，配送侧需优先保障"
        write_csv(high, PROCESSED_DIR / "high_priority_products.csv")

    if find_file("outlet_metrics.csv") is None:
        rows = []
        for idx in range(1, 61):
            risk_level = "高风险" if idx <= 9 else "中风险" if idx <= 26 else "低风险"
            rows.append(
                {
                    "outlet_id": f"O{idx:03d}",
                    "outlet_name": f"演示网点{idx:02d}",
                    "town": "乡镇片区" if idx <= 42 else "城区街道",
                    "region_type": "乡镇" if idx <= 42 else "城区",
                    "lon": 115.20 + (idx % 15) * 0.025,
                    "lat": 35.00 + (idx // 15) * 0.08,
                    "road_level": 2 if idx <= 9 else 4,
                    "service_level": 2 if idx <= 9 else 4,
                    "order_count_90d": 90 + idx,
                    "timeout_rate_calc": 0.18 if risk_level == "高风险" else 0.08,
                    "avg_actual_hours": 42 if risk_level == "高风险" else 28,
                    "return_rate_calc": 0.06,
                    "risk_level": risk_level,
                }
            )
        write_csv(pd.DataFrame(rows), PROCESSED_DIR / "outlet_metrics.csv")

    if find_file("delivery_exception_result.csv") is None:
        rows = []
        type_plan = (
            ["A类重点商品超时订单"] * m["a_timeout_orders"]
            + ["生鲜冷链超时订单"] * m["fresh_timeout_orders"]
            + ["超长配送订单"] * m["long_timeout_orders"]
            + ["乡镇偏远超时订单"] * m["township_timeout_orders"]
        )
        type_plan += ["普通超时订单"] * max(m["exception_orders"] - len(type_plan), 0)
        for idx, exception_type in enumerate(type_plan, start=1):
            rows.append(
                {
                    "delivery_id": f"D{idx:06d}",
                    "order_id": f"ORD{idx:06d}",
                    "product_id": f"P{(idx % 200) + 1:04d}",
                    "product_name": f"重点商品{idx % 40:02d}",
                    "abc_class": "A" if "A类" in exception_type else "B",
                    "category": "冷链食品" if "生鲜" in exception_type else "日用品",
                    "outlet_id": f"O{(idx % 60) + 1:03d}",
                    "actual_hours_calc": 78 if "超长" in exception_type else 44,
                    "is_timeout": 1,
                    "exception_type": exception_type,
                    "exception_reason": "配送时效超过业务阈值",
                    "priority_level": "高" if "A类" in exception_type or "生鲜" in exception_type else "中",
                    "suggest_action": "优先调度，必要时纳入低空应急配送",
                }
            )
        write_csv(pd.DataFrame(rows), PROCESSED_DIR / "delivery_exception_result.csv")

    if find_file("outlet_exception_summary.csv") is None:
        rows = []
        for idx in range(1, 61):
            exception_count = 12 if idx <= 9 else 6 if idx <= 26 else 2
            rows.append(
                {
                    "outlet_id": f"O{idx:03d}",
                    "total_delivery_orders": 80,
                    "normal_timeout_orders": max(exception_count - 3, 0),
                    "long_timeout_orders": 2 if idx <= 9 else 1,
                    "a_class_timeout_orders": 3 if idx <= 9 else 1,
                    "fresh_cold_timeout_orders": 2 if idx <= 9 else 0,
                    "township_timeout_orders": 4 if idx <= 42 else 0,
                    "high_priority_timeout_orders": 4 if idx <= 9 else 1,
                    "exception_order_count": exception_count,
                    "exception_rate": round(exception_count / 80, 4),
                    "serious_exception_count": 5 if idx <= 9 else 1,
                    "exception_score_base": 82 if idx <= 9 else 55,
                }
            )
        write_csv(pd.DataFrame(rows), PROCESSED_DIR / "outlet_exception_summary.csv")

    if find_file("outlet_risk_score.csv") is None:
        outlet = read_csv(find_file("outlet_metrics.csv") or PROCESSED_DIR / "outlet_metrics.csv")
        risk_levels = ["高风险"] * 9 + ["中风险"] * 17 + ["低风险"] * 34
        outlet = outlet.head(60).copy()
        outlet["exception_order_count"] = [12 if i < 9 else 6 if i < 26 else 2 for i in range(len(outlet))]
        outlet["exception_rate"] = (outlet["exception_order_count"] / 80).round(4)
        outlet["a_class_timeout_orders"] = [3 if i < 9 else 1 for i in range(len(outlet))]
        outlet["fresh_cold_timeout_orders"] = [2 if i < 9 else 0 for i in range(len(outlet))]
        outlet["long_timeout_orders"] = [2 if i < 9 else 1 for i in range(len(outlet))]
        outlet["high_priority_timeout_orders"] = [4 if i < 9 else 1 for i in range(len(outlet))]
        outlet["final_risk_score"] = [82 if item == "高风险" else 62 if item == "中风险" else 35 for item in risk_levels]
        outlet["risk_score"] = outlet["final_risk_score"]
        outlet["risk_level"] = risk_levels
        outlet["main_risk_reason"] = outlet["risk_level"].map({"高风险": "A类与生鲜订单超时集中", "中风险": "配送时长偏高", "低风险": "运行平稳"})
        write_csv(outlet, PROCESSED_DIR / "outlet_risk_score.csv")

    if find_file("cluster_result.csv") is None:
        risk = read_csv(find_file("outlet_risk_score.csv") or PROCESSED_DIR / "outlet_risk_score.csv")
        risk["cluster_id"] = [idx % 4 for idx in range(len(risk))]
        risk["cluster_name"] = risk["cluster_id"].map(lambda item: f"配送优化片区{int(item) + 1}")
        write_csv(risk, PROCESSED_DIR / "cluster_result.csv")

    if find_file("launch_candidates.csv") is None:
        rows = []
        for cluster_id in range(4):
            for rank in range(1, 3):
                idx = cluster_id * 2 + rank
                rows.append(
                    {
                        "candidate_id": f"LC{idx:03d}",
                        "candidate_name": f"片区{cluster_id}候选起降点{rank}",
                        "outlet_id": f"O{idx:03d}",
                        "cluster_id": cluster_id,
                        "candidate_rank_in_cluster": rank,
                        "candidate_label": f"片区{cluster_id}-候选{rank}",
                        "town": "乡镇片区",
                        "lon": 115.25 + cluster_id * 0.08 + rank * 0.01,
                        "lat": 35.05 + cluster_id * 0.08 + rank * 0.01,
                        "road_level": 3,
                        "service_level": 3,
                        "risk_score": 80 - rank,
                        "high_priority_order_count": 5,
                        "service_radius_km": 5,
                        "candidate_score": 90 - rank,
                    }
                )
        write_csv(pd.DataFrame(rows), PROCESSED_DIR / "launch_candidates.csv")

    if find_file("launch_points.csv") is None:
        candidates = read_csv(find_file("launch_candidates.csv") or PROCESSED_DIR / "launch_candidates.csv")
        points = candidates.sort_values("candidate_rank_in_cluster").groupby("cluster_id", as_index=False).head(1).head(3).copy()
        points["launch_point_id"] = [f"LP{idx + 1:03d}" for idx in range(len(points))]
        points["launch_point_name"] = points["candidate_name"].str.replace("候选", "推荐", regex=False)
        write_csv(points, PROCESSED_DIR / "launch_points.csv")

    if find_file("route_candidates.csv") is None:
        write_csv(
            pd.DataFrame(
                [
                    ("R-A", "最短直达航线", 4.8, 13, "中", 84.2, "时效优先"),
                    ("R-B", "低风险绕行航线", 5.6, 16, "低", 88.5, "综合安全得分最高"),
                    ("R-C", "多网点覆盖航线", 6.4, 20, "中", 80.6, "覆盖能力较强"),
                    ("R-R", "备用返航航线", 5.9, 18, "低", 78.4, "异常情况下可返航"),
                ],
                columns=["route_id", "route_type", "distance_km", "estimated_time_min", "risk_level", "reward_score", "recommend_reason"],
            ),
            PROCESSED_DIR / "route_candidates.csv",
        )

    if find_file("route_assignment.csv") is None:
        rows = []
        for idx in range(1, 29):
            rows.append(
                {
                    "order_id": f"ORD{idx:06d}",
                    "product_id": f"P{idx:04d}",
                    "product_name": f"高优先级商品{idx:02d}",
                    "abc_class": "A",
                    "category": "冷链食品" if idx <= 12 else "生鲜粮油",
                    "outlet_id": f"O{idx % 60 + 1:03d}",
                    "launch_point_id": f"LP{idx % 3 + 1:03d}",
                    "route_id": f"R-B-{idx:03d}",
                    "route_type": "低风险绕行航线",
                    "estimated_distance_km": 5.2,
                    "estimated_time_min": 16,
                    "risk_level": "低" if idx <= 22 else "不可行",
                    "reward_score": 88.5 if idx <= 22 else 0,
                    "assigned_reason": "适合A类重点商品或生鲜冷链异常订单的低空应急配送" if idx <= 22 else "距离或天气约束下暂不建议低空配送",
                }
            )
        write_csv(pd.DataFrame(rows), PROCESSED_DIR / "route_assignment.csv")

    if find_file("route_simulation_report.html") is None:
        write_text(
            "<html><meta charset='utf-8'><body><h1>低空航线仿真报告</h1><p>推荐航线：低风险绕行航线。</p></body></html>",
            REPORT_DIR / "route_simulation_report.html",
        )


def build_metrics() -> dict[str, float | int]:
    metrics = dict(DEFAULT_METRICS)

    base_path = find_file("product_base_metrics.csv")
    if base_path:
        base = read_csv(base_path)
        metrics["product_total"] = int(base["product_id"].nunique()) if "product_id" in base.columns else len(base)

    abc_summary_path = find_file("abc_summary.csv")
    if abc_summary_path:
        abc_summary = read_csv(abc_summary_path)
        for cls, key in [("A", "a_products"), ("B", "b_products"), ("C", "c_products")]:
            part = abc_summary[abc_summary["abc_class"].astype(str).eq(cls)] if "abc_class" in abc_summary.columns else pd.DataFrame()
            if not part.empty and "product_count" in part.columns:
                metrics[key] = int(to_number(part["product_count"]).iloc[0])
        for cls, key in [("A", "a_sales_rate"), ("B", "b_sales_rate"), ("C", "c_sales_rate")]:
            part = abc_summary[abc_summary["abc_class"].astype(str).eq(cls)] if "abc_class" in abc_summary.columns else pd.DataFrame()
            if not part.empty and "sales_amount_rate" in part.columns:
                value = float(to_number(part["sales_amount_rate"]).iloc[0])
                metrics[key] = round(value * 100 if value <= 1 else value, 1)
        c_part = abc_summary[abc_summary["abc_class"].astype(str).eq("C")] if "abc_class" in abc_summary.columns else pd.DataFrame()
        if not c_part.empty and "stock_amount_rate" in c_part.columns:
            value = float(to_number(c_part["stock_amount_rate"]).iloc[0])
            metrics["c_stock_rate"] = round(value * 100 if value <= 1 else value, 1)

    warning_path = find_file("inventory_warning.csv")
    if warning_path:
        warning = read_csv(warning_path)
        if "inventory_risk" in warning.columns:
            counts = warning["inventory_risk"].value_counts()
            metrics["shortage_products"] = int(counts.get("缺货风险", metrics["shortage_products"]))
            metrics["normal_stock_products"] = int(counts.get("库存正常", metrics["normal_stock_products"]))
            metrics["overstock_products"] = int(counts.get("库存积压", metrics["overstock_products"]))
            metrics["high_overstock_products"] = int(counts.get("高库存积压", metrics["high_overstock_products"]))

    purchase_path = find_file("purchase_advice.csv")
    if purchase_path:
        purchase = read_csv(purchase_path)
        metrics["shortage_products"] = len(purchase) or metrics["shortage_products"]
        amount_cols = [col for col in ["suggest_purchase_amount", "purchase_amount"] if col in purchase.columns]
        if amount_cols:
            metrics["purchase_amount"] = int(to_number(purchase[amount_cols[0]]).sum())

    clearance_path = find_file("clearance_advice.csv")
    if clearance_path:
        clearance = read_csv(clearance_path)
        metrics["high_overstock_products"] = len(clearance) or metrics["high_overstock_products"]
        amount_cols = [col for col in ["expected_release_amount", "release_amount", "stock_amount"] if col in clearance.columns]
        if amount_cols:
            total = int(to_number(clearance[amount_cols[0]]).sum())
            metrics["release_amount"] = total if total > 0 else metrics["release_amount"]

    delivery_path = find_file("clean_delivery.csv")
    if delivery_path:
        delivery = read_csv(delivery_path)
        metrics["delivery_records"] = len(delivery)

    exception_path = find_file("delivery_exception_result.csv")
    if exception_path:
        exceptions = read_csv(exception_path)
        metrics["exception_orders"] = len(exceptions)
        if "exception_type" in exceptions.columns:
            text = exceptions["exception_type"].astype(str)
            metrics["a_timeout_orders"] = int(text.str.contains("A类", na=False).sum()) or metrics["a_timeout_orders"]
            metrics["fresh_timeout_orders"] = int(text.str.contains("生鲜|冷链", regex=True, na=False).sum()) or metrics["fresh_timeout_orders"]
            metrics["long_timeout_orders"] = int(text.str.contains("超长", na=False).sum()) or metrics["long_timeout_orders"]
            metrics["township_timeout_orders"] = int(text.str.contains("乡镇|偏远", regex=True, na=False).sum()) or metrics["township_timeout_orders"]

    outlet_path = find_file("outlet_risk_score.csv") or find_file("outlet_metrics.csv")
    if outlet_path:
        outlets = read_csv(outlet_path)
        metrics["outlet_total"] = int(outlets["outlet_id"].nunique()) if "outlet_id" in outlets.columns else len(outlets)
        if "risk_level" in outlets.columns:
            counts = outlets["risk_level"].astype(str).value_counts()
            metrics["high_risk_outlets"] = int(counts.get("高风险", metrics["high_risk_outlets"]))
            metrics["medium_risk_outlets"] = int(counts.get("中风险", metrics["medium_risk_outlets"]))
            metrics["low_risk_outlets"] = int(counts.get("低风险", metrics["low_risk_outlets"]))
            if "region_type" in outlets.columns:
                metrics["township_high_risk_outlets"] = int(((outlets["risk_level"].astype(str) == "高风险") & (outlets["region_type"].astype(str) == "乡镇")).sum()) or metrics["township_high_risk_outlets"]

    cluster_path = find_file("cluster_result.csv")
    if cluster_path:
        clusters = read_csv(cluster_path)
        if "cluster_id" in clusters.columns:
            metrics["cluster_count"] = int(clusters["cluster_id"].nunique())

    candidates_path = find_file("launch_candidates.csv")
    if candidates_path:
        metrics["launch_candidates"] = len(read_csv(candidates_path))

    launch_path = find_file("launch_points.csv")
    if launch_path:
        metrics["launch_points"] = len(read_csv(launch_path))

    route_path = find_file("route_assignment.csv")
    if route_path:
        routes = read_csv(route_path)
        metrics["route_planned_orders"] = len(routes)
        if "risk_level" in routes.columns:
            covered = int((routes["risk_level"].astype(str) != "不可行").sum())
            metrics["route_covered_orders"] = covered or min(metrics["route_planned_orders"], metrics["route_covered_orders"])
        else:
            metrics["route_covered_orders"] = min(metrics["route_planned_orders"], metrics["route_covered_orders"])
        metrics["route_uncovered_orders"] = max(int(metrics["route_planned_orders"]) - int(metrics["route_covered_orders"]), 0)

    return metrics


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(min(len(items), 5))
    for idx, (label, value, note) in enumerate(items):
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>{html.escape(label)}</span>
                    <strong>{html.escape(value)}</strong>
                    <p>{html.escape(note)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_plan_cards(plans: pd.DataFrame) -> None:
    if plans.empty:
        return
    cards = []
    for _, row in plans.iterrows():
        is_top = int(row.get("ai_recommendation", 0)) == 1
        border = "#38bdf8" if is_top else "#2a3342"
        badge_bg = "#0ea5e9" if is_top else "#334155"
        badge_text = "AI推荐" if is_top else f"方案{int(row.get('ai_recommendation', 0))}"
        score = html.escape(str(row.get("ai_score", "")))
        plan_name = html.escape(str(row.get("plan_name", "")))
        position = html.escape(str(row.get("plan_position", "")))
        core_actions = html.escape(str(row.get("core_actions", "")))
        expected_effect = html.escape(str(row.get("expected_effect", "")))
        estimated_cost = html.escape(str(row.get("estimated_cost", "")))
        risk_level = html.escape(str(row.get("risk_level", "")))
        try:
            score_width = min(float(row.get("ai_score", 0)), 100)
        except (TypeError, ValueError):
            score_width = 0
        cards.append(
            f'<div style="border:1px solid {border}; border-radius:10px; padding:14px 16px; background:#111827; min-height:224px;">'
            f'<div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">'
            f'<div><span style="display:inline-block; padding:3px 9px; border-radius:999px; background:{badge_bg}; color:#fff; font-size:12px;">{badge_text}</span>'
            f'<h4 style="margin:10px 0 4px; color:#f8fafc; font-size:18px;">{plan_name}</h4>'
            f'<p style="margin:0; color:#cbd5e1; font-size:13px;">{position}</p></div>'
            f'<div style="text-align:right;"><div style="color:#93c5fd; font-size:12px;">AI评分</div>'
            f'<strong style="color:#f8fafc; font-size:28px;">{score}</strong></div></div>'
            f'<div style="height:8px; background:#1f2937; border-radius:999px; margin:14px 0 12px;">'
            f'<div style="height:8px; width:{score_width}%; background:#38bdf8; border-radius:999px;"></div></div>'
            f'<p style="margin:0 0 10px; color:#f8fafc; line-height:1.55;"><strong>核心措施：</strong>{core_actions}</p>'
            f'<p style="margin:0 0 12px; color:#cbd5e1; line-height:1.55;"><strong>预计效果：</strong>{expected_effect}</p>'
            f'<div style="display:flex; gap:8px; flex-wrap:wrap;">'
            f'<span style="padding:4px 8px; border-radius:6px; background:#1f2937; color:#e2e8f0; font-size:12px;">成本：{estimated_cost}</span>'
            f'<span style="padding:4px 8px; border-radius:6px; background:#1f2937; color:#e2e8f0; font-size:12px;">风险：{risk_level}</span>'
            f'</div></div>'
        )
    st.markdown(
        f'<div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin-top:6px;">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_effect_cards(effects: pd.DataFrame) -> None:
    if effects.empty:
        return
    cards = []
    for _, row in effects.iterrows():
        kpi_name = html.escape(str(row.get("kpi_name", "")))
        current_value = html.escape(str(row.get("current_value", "")))
        improvement = html.escape(str(row.get("expected_improvement", "")))
        desc = html.escape(str(row.get("improvement_desc", "")))
        cards.append(
            f'<div style="border:1px solid #273244; border-radius:10px; padding:14px 16px; background:#111827; min-height:132px;">'
            f'<div style="color:#cbd5e1; font-size:13px; margin-bottom:8px;">{kpi_name}</div>'
            f'<div style="display:flex; align-items:baseline; justify-content:space-between; gap:10px;">'
            f'<div><span style="color:#94a3b8; font-size:12px;">当前</span><div style="color:#f8fafc; font-size:20px; font-weight:700;">{current_value}</div></div>'
            f'<div style="text-align:right;"><span style="color:#93c5fd; font-size:12px;">预计改善</span><div style="color:#38bdf8; font-size:24px; font-weight:800;">{improvement}</div></div>'
            f'</div>'
            f'<div style="margin-top:10px; padding-top:10px; border-top:1px solid #1f2937; color:#cbd5e1; font-size:12px; line-height:1.5;">{desc}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; margin:8px 0 14px;">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def set_next_page(page: str) -> None:
    if page in PAGES:
        st.session_state["module4_next_page"] = page


def get_report_prompt() -> str:
    return st.session_state.get("module4_prompt_text", DEFAULT_PROMPT)


def save_prompt(prompt: str, report_type: str) -> Path:
    text = (
        f"报告类型：{report_type}\n"
        f"保存时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"{prompt.strip()}\n"
    )
    path = write_text(text, REPORT_DIR / "llm_prompt_record.txt")
    write_log("保存报告提示词", "prompt_text", "llm_prompt_record.txt", remark=report_type)
    return path


def optimize_report_prompt(simple_requirement: str, report_type: str) -> str:
    requirement = simple_requirement.strip() or "生成一份企业经营诊断报告"
    return f"""请你作为物流企业经营决策顾问，基于前三个模块输出结果和RAG经营知识库，生成《{report_type}》。

用户原始需求：
{requirement}

报告生成要求：
1. 必须使用模块一的数据质量报告、清洗订单、商品基础指标和网点基础指标；
2. 必须使用模块二的ABC分类、销量预测、库存预警、补货建议和清仓建议；
3. 必须使用模块三的配送异常、网点风险评分、K-Means片区、起降点和低空航线规划结果；
4. 不允许编造未在数据中出现的指标；
5. 从库存结构、缺货风险、配送异常、网点风险、低空应急配送五个角度分析；
6. 输出高优先级和中优先级经营建议；
7. 报告面向企业管理层，语言要简洁、正式、可执行；
8. 最后给出部门行动建议和预期经营价值。"""


def mock_answer(question: str, metrics: dict[str, float | int]) -> str:
    q = question.strip() or "企业下一步应该优先做哪些调整？"
    if "利润" in q:
        return (
            f"根据系统分析，利润下降主要来自四个方面：第一，C类商品库存占用偏高。C类商品有{metrics['c_products']}种，"
            f"销售贡献约{metrics['c_sales_rate']}%，但库存金额占比达到{metrics['c_stock_rate']}%，造成资金占用。第二，A类商品存在缺货风险，"
            f"A类商品{metrics['a_products']}种，贡献约{metrics['a_sales_rate']}%销售额，一旦缺货会影响核心销售。第三，配送异常影响履约质量，"
            f"系统识别异常配送订单{metrics['exception_orders']}条，其中A类重点商品超时{metrics['a_timeout_orders']}条，生鲜冷链超时{metrics['fresh_timeout_orders']}条。"
            f"第四，高风险网点主要集中在乡镇区域，高风险网点{metrics['high_risk_outlets']}个，其中乡镇高风险网点{metrics['township_high_risk_outlets']}个。"
            "因此企业应同步推进A类商品补货、C类商品清仓、乡镇网点调度优化和低空应急配送试点。"
        )
    if "补货" in q:
        return (
            f"优先补货对象应聚焦A类商品和缺货风险商品。当前A类商品{metrics['a_products']}种，贡献约{metrics['a_sales_rate']}%销售额，"
            f"系统识别缺货风险商品{metrics['shortage_products']}种，建议补货金额约{metrics['purchase_amount']}元。"
            "建议采购部门先保障A类民生商品、生鲜粮油和冷链食品，再根据未来30天销量预测补充B类稳定销售商品。"
        )
    if "清仓" in q:
        return (
            f"清仓对象应聚焦C类高库存商品。当前C类商品{metrics['c_products']}种，销售贡献约{metrics['c_sales_rate']}%，"
            f"但库存金额占比达到{metrics['c_stock_rate']}%。系统建议清仓商品{metrics['high_overstock_products']}种，预计释放库存资金{metrics['release_amount']}元。"
            "建议采用降价促销、组合销售、停止采购和仓位调整等方式降低库存占用。"
        )
    if "网点" in q or "配送风险" in q:
        return (
            f"配送风险最高的网点主要集中在乡镇和边缘片区。系统识别高风险网点{metrics['high_risk_outlets']}个，"
            f"其中乡镇高风险网点{metrics['township_high_risk_outlets']}个。风险来源包括平均配送时长偏高、历史超时率高、道路等级低、服务能力不足，"
            f"以及A类商品和生鲜冷链订单超时集中。建议将这些网点优先纳入片区调度优化和低空应急配送试点。"
        )
    if "起降点" in q:
        return (
            f"规划低空应急起降点是为了解决乡镇高风险区域的重点商品配送问题。系统已将{metrics['outlet_total']}个网点划分为"
            f"{metrics['cluster_count']}个配送优化片区，并筛选{metrics['launch_candidates']}个候选起降点，推荐{metrics['launch_points']}个起降点用于试点。"
            "起降点不是替代普通配送，而是服务A类重点商品、生鲜冷链商品和高风险异常订单。"
        )
    if "低空航线" in q or "订单" in q:
        return (
            f"低空航线应优先服务A类重点商品、生鲜冷链商品和乡镇高风险异常订单。当前系统纳入低空规划订单{metrics['route_planned_orders']}单，"
            f"其中可低空覆盖{metrics['route_covered_orders']}单，不建议低空配送{metrics['route_uncovered_orders']}单。"
            "普通订单仍以地面配送为主，低空资源用于高价值、高时效、高风险场景。"
        )
    if "库存积压" in q or "配送超时" in q:
        return (
            "降低库存积压和配送超时需要同时从库存侧和履约侧行动。库存侧优先补货A类缺货商品，清仓C类高库存商品；"
            f"配送侧优先处理{metrics['a_timeout_orders']}条A类重点商品超时订单和{metrics['fresh_timeout_orders']}条生鲜冷链超时订单，"
            f"并把{metrics['high_risk_outlets']}个高风险网点纳入片区调度和低空应急试点。"
        )
    return (
        "企业下一步建议按优先级推进四件事：第一，补货A类缺货商品，保障核心销售；第二，清仓C类高库存商品，释放库存资金；"
        "第三，优先处理A类和生鲜冷链超时订单，提升履约质量；第四，在乡镇高风险片区试点低空应急配送，逐步形成县域物流低空配送示范。"
    )


def save_answer_record(question: str, answer: str) -> Path:
    text = (
        f"问题：{question}\n"
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"模拟回答：\n{answer}\n"
    )
    path = write_text(text, REPORT_DIR / "llm_response_mock.txt")
    write_log("保存大模型模拟回答", "经营问题", "llm_response_mock.txt")
    return path


def management_summary_df(metrics: dict[str, float | int]) -> pd.DataFrame:
    rows = [
        ("商品总数", metrics["product_total"], "商品经营"),
        ("A类商品", metrics["a_products"], "库存重点保障"),
        ("C类商品", metrics["c_products"], "库存优化关注"),
        ("缺货风险商品", metrics["shortage_products"], "采购补货"),
        ("建议清仓商品", metrics["high_overstock_products"], "库存释放"),
        ("异常配送订单", metrics["exception_orders"], "配送履约"),
        ("A类重点商品超时", metrics["a_timeout_orders"], "重点保障"),
        ("生鲜冷链超时", metrics["fresh_timeout_orders"], "质量风险"),
        ("高风险网点", metrics["high_risk_outlets"], "网点治理"),
        ("配送优化片区", metrics["cluster_count"], "K-Means聚类"),
        ("候选起降点", metrics["launch_candidates"], "低空规划"),
        ("推荐起降点", metrics["launch_points"], "试点建设"),
        ("纳入低空规划订单", metrics["route_planned_orders"], "高优先级异常订单"),
        ("可低空覆盖订单", metrics["route_covered_orders"], "应急配送能力"),
    ]
    return pd.DataFrame(rows, columns=["metric_name", "metric_value", "business_area"])


def decision_actions_df(metrics: dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("高", "补货A类缺货商品", "采购与库存部门", f"缺货风险商品{metrics['shortage_products']}种，A类贡献{metrics['a_sales_rate']}%销售额", "立即执行"),
            ("高", "优先处理A类和生鲜冷链超时订单", "配送调度部门", f"A类超时{metrics['a_timeout_orders']}条，生鲜冷链超时{metrics['fresh_timeout_orders']}条", "立即执行"),
            ("高", "在乡镇高风险片区试点低空应急配送", "运营与低空项目组", f"高风险网点{metrics['high_risk_outlets']}个，候选起降点{metrics['launch_candidates']}个", "试点推进"),
            ("中", "清仓C类高库存商品", "销售与仓储部门", f"C类库存金额占比{metrics['c_stock_rate']}%，预计释放{metrics['release_amount']}元", "两周内启动"),
            ("中", "调整乡镇网点服务能力", "网点运营部门", f"乡镇高风险网点{metrics['township_high_risk_outlets']}个", "持续优化"),
        ],
        columns=["priority", "action_item", "owner_department", "data_basis", "execution_status"],
    )


def decision_action_candidates(metrics: dict[str, float | int]) -> pd.DataFrame:
    rows = [
        {
            "action_item": "补货A类缺货商品",
            "owner_department": "采购部",
            "sales_score": 95,
            "inventory_score": 72,
            "delivery_score": 48,
            "feasibility_score": 82,
            "risk_score": 86,
            "data_basis": f"A类商品{metrics['a_products']}种，缺货风险商品{metrics['shortage_products']}种",
            "ai_reason": "直接保障核心销售商品，优先级最高。",
        },
        {
            "action_item": "清仓C类高库存商品",
            "owner_department": "仓储部",
            "sales_score": 55,
            "inventory_score": 96,
            "delivery_score": 36,
            "feasibility_score": 78,
            "risk_score": 74,
            "data_basis": f"C类商品{metrics['c_products']}种，库存金额占比{metrics['c_stock_rate']}%",
            "ai_reason": "释放库存资金，降低仓储占用。",
        },
        {
            "action_item": "优先处理A类和生鲜冷链超时订单",
            "owner_department": "配送部",
            "sales_score": 88,
            "inventory_score": 46,
            "delivery_score": 94,
            "feasibility_score": 84,
            "risk_score": 90,
            "data_basis": f"A类超时{metrics['a_timeout_orders']}条，生鲜冷链超时{metrics['fresh_timeout_orders']}条",
            "ai_reason": "减少重点商品履约风险和客户投诉。",
        },
        {
            "action_item": "治理乡镇高风险网点",
            "owner_department": "网点运营部",
            "sales_score": 64,
            "inventory_score": 42,
            "delivery_score": 91,
            "feasibility_score": 70,
            "risk_score": 88,
            "data_basis": f"高风险网点{metrics['high_risk_outlets']}个，乡镇高风险网点{metrics['township_high_risk_outlets']}个",
            "ai_reason": "提升县域末端配送稳定性。",
        },
        {
            "action_item": "启动低空应急配送试点",
            "owner_department": "低空项目组",
            "sales_score": 70,
            "inventory_score": 30,
            "delivery_score": 89,
            "feasibility_score": 62,
            "risk_score": 82,
            "data_basis": f"候选起降点{metrics['launch_candidates']}个，可低空覆盖订单{metrics['route_covered_orders']}单",
            "ai_reason": "用于A类、生鲜冷链和乡镇异常订单的应急保障。",
        },
        {
            "action_item": "建立周度经营复盘机制",
            "owner_department": "管理层",
            "sales_score": 62,
            "inventory_score": 68,
            "delivery_score": 66,
            "feasibility_score": 92,
            "risk_score": 76,
            "data_basis": "整合库存、配送、网点和低空试点指标",
            "ai_reason": "把一次性分析固化为持续经营管理机制。",
        },
    ]
    return pd.DataFrame(rows)


def score_decision_actions(
    metrics: dict[str, float | int],
    goal: str,
    weights: dict[str, int],
    selected_departments: list[str],
    cycle: str,
) -> pd.DataFrame:
    df = decision_action_candidates(metrics)
    if selected_departments:
        df = df[df["owner_department"].isin(selected_departments)].copy()
    if df.empty:
        return df
    total_weight = max(sum(weights.values()), 1)
    df["ai_score"] = (
        df["sales_score"] * weights["sales"]
        + df["inventory_score"] * weights["inventory"]
        + df["delivery_score"] * weights["delivery"]
        + df["feasibility_score"] * weights["feasibility"]
        + df["risk_score"] * weights["risk"]
    ) / total_weight
    goal_bonus = {
        "优先提升销售": {"sales_score": 8},
        "优先降低库存占用": {"inventory_score": 8},
        "优先降低配送超时": {"delivery_score": 8},
        "优先保障乡镇网点": {"delivery_score": 5, "risk_score": 5},
        "优先推进低空配送试点": {"delivery_score": 4, "risk_score": 4},
    }
    for column, bonus in goal_bonus.get(goal, {}).items():
        df["ai_score"] += (df[column] / 100) * bonus
    cycle_note = {
        "7天应急处理": "立即执行",
        "15天专项优化": "两周内完成",
        "30天经营提升": "月度推进",
    }
    df["ai_score"] = df["ai_score"].round(1)
    df["priority_rank"] = df["ai_score"].rank(method="first", ascending=False).astype(int)
    df["execution_cycle"] = cycle_note.get(cycle, "月度推进")
    df["priority"] = pd.cut(
        df["ai_score"],
        bins=[0, 72, 84, 120],
        labels=["中", "高", "最高"],
        include_lowest=True,
    ).astype(str)
    result = df.sort_values("ai_score", ascending=False)
    return result[
        [
            "priority_rank",
            "priority",
            "action_item",
            "owner_department",
            "data_basis",
            "ai_score",
            "execution_cycle",
            "ai_reason",
        ]
    ]


def department_tasks_df(actions: pd.DataFrame, cycle: str) -> pd.DataFrame:
    task_deadline = {
        "7天应急处理": "7天内",
        "15天专项优化": "15天内",
        "30天经营提升": "30天内",
    }.get(cycle, "30天内")
    rows = []
    for _, row in actions.head(5).iterrows():
        rows.append(
            {
                "department": row["owner_department"],
                "task": row["action_item"],
                "deadline": task_deadline,
                "data_basis": row["data_basis"],
                "expected_output": "形成执行记录并回写经营报告",
            }
        )
    return pd.DataFrame(rows)


def simulate_improvement(metrics: dict[str, float | int], actions: pd.DataFrame, cycle: str) -> pd.DataFrame:
    action_text = " ".join(actions["action_item"].astype(str).tolist()) if not actions.empty else ""
    cycle_factor = {"7天应急处理": 0.45, "15天专项优化": 0.7, "30天经营提升": 1.0}.get(cycle, 1.0)
    release_amount = int(metrics["release_amount"] * (0.35 if "清仓" in action_text else 0.12) * cycle_factor)
    timeout_reduction = int((metrics["a_timeout_orders"] + metrics["fresh_timeout_orders"] + metrics["township_timeout_orders"]) * (0.32 if "超时" in action_text or "网点" in action_text else 0.12) * cycle_factor)
    shortage_reduction = int(metrics["shortage_products"] * (0.45 if "补货" in action_text else 0.15) * cycle_factor)
    low_altitude_orders = int(metrics["route_covered_orders"] * (0.55 if "低空" in action_text else 0.18) * cycle_factor)
    rows = [
        ("预计释放库存资金", release_amount, f"{release_amount}元", "来自C类高库存商品清仓"),
        ("预计减少重点超时订单", timeout_reduction, f"{timeout_reduction}条", "来自配送调度和网点治理"),
        ("预计降低缺货商品数", shortage_reduction, f"{shortage_reduction}种", "来自A类商品补货"),
        ("预计低空应急覆盖订单", low_altitude_orders, f"{low_altitude_orders}单", "来自低空试点航线"),
    ]
    return pd.DataFrame(rows, columns=["effect_metric", "numeric_value", "expected_value", "reason"])


def problem_data_basis(metrics: dict[str, float | int], problem: str) -> str:
    if problem == "库存资金占用过高":
        return f"C类商品{metrics['c_products']}种，库存金额占比{metrics['c_stock_rate']}%，建议清仓商品{metrics['high_overstock_products']}种。"
    if problem == "A类商品缺货风险":
        return f"A类商品{metrics['a_products']}种，贡献{metrics['a_sales_rate']}%销售额，缺货风险商品{metrics['shortage_products']}种。"
    if problem == "乡镇配送超时严重":
        return f"异常配送订单{metrics['exception_orders']}条，高风险网点{metrics['high_risk_outlets']}个，乡镇高风险网点{metrics['township_high_risk_outlets']}个。"
    if problem == "生鲜冷链订单履约风险":
        return f"生鲜冷链超时{metrics['fresh_timeout_orders']}条，A类重点商品超时{metrics['a_timeout_orders']}条，超长配送{metrics['long_timeout_orders']}条。"
    return f"候选起降点{metrics['launch_candidates']}个，推荐起降点{metrics['launch_points']}个，可低空覆盖订单{metrics['route_covered_orders']}单。"


def plan_core_actions(problem: str) -> dict[str, str]:
    action_map = {
        "库存资金占用过高": {
            "保守方案": "暂停C类商品补采，先处理库龄最长和周转最慢商品。",
            "均衡方案": "A类商品保障补货，C类商品组合促销，前置仓库存同步调整。",
            "激进方案": "集中清仓C类高库存商品，释放仓容和库存资金。",
        },
        "A类商品缺货风险": {
            "保守方案": "仅对安全库存以下的A类商品补货。",
            "均衡方案": "结合销量预测对A类商品分批补货，并设置补货上限。",
            "激进方案": "扩大A类重点商品采购量，优先保障核心销售。",
        },
        "乡镇配送超时严重": {
            "保守方案": "优先调整高风险乡镇网点派车顺序。",
            "均衡方案": "高风险网点加车加班，重点订单纳入低空试点评估。",
            "激进方案": "集中治理乡镇高风险网点，并快速启动低空应急配送试点。",
        },
        "生鲜冷链订单履约风险": {
            "保守方案": "对生鲜冷链超时订单建立人工预警。",
            "均衡方案": "生鲜冷链订单优先分拣、优先派送，异常网点重点调度。",
            "激进方案": "将A类冷链和生鲜急件纳入低空应急配送优先队列。",
        },
        "低空应急配送试点评估": {
            "保守方案": "选择1个推荐起降点开展单线路验证。",
            "均衡方案": "选择2到3个起降点覆盖高风险片区，控制试点范围。",
            "激进方案": "按4个聚类片区同步试点，优先覆盖A类和生鲜冷链异常订单。",
        },
    }
    return action_map.get(problem, action_map["乡镇配送超时严重"])


def generate_decision_plans(
    metrics: dict[str, float | int],
    problem: str,
    budget_level: str,
    cycle: str,
    risk_preference: str,
    departments: list[str],
    weights: dict[str, int],
) -> pd.DataFrame:
    actions = plan_core_actions(problem)
    dept_text = "、".join(departments) if departments else "全部相关部门"
    basis = problem_data_basis(metrics, problem)
    base_rows = [
        {
            "plan_name": "保守方案",
            "plan_position": "成本低、执行快、风险小",
            "core_actions": actions["保守方案"],
            "estimated_cost": "低",
            "execution_difficulty": "低",
            "risk_level": "低",
            "expected_effect": "短期改善明显，适合先稳住经营风险。",
            "recommended_departments": dept_text,
            "sales_score": 68,
            "inventory_score": 72,
            "delivery_score": 66,
            "cost_score": 92,
            "risk_score": 90,
        },
        {
            "plan_name": "均衡方案",
            "plan_position": "库存、配送、成本综合优化",
            "core_actions": actions["均衡方案"],
            "estimated_cost": "中",
            "execution_difficulty": "中",
            "risk_level": "中",
            "expected_effect": "兼顾销售保障、库存释放和配送履约，适合正式采用。",
            "recommended_departments": dept_text,
            "sales_score": 82,
            "inventory_score": 84,
            "delivery_score": 86,
            "cost_score": 78,
            "risk_score": 82,
        },
        {
            "plan_name": "激进方案",
            "plan_position": "投入更高、改善更快、试点力度更大",
            "core_actions": actions["激进方案"],
            "estimated_cost": "高",
            "execution_difficulty": "高",
            "risk_level": "高",
            "expected_effect": "对重点问题改善最快，但需要更强预算和组织配合。",
            "recommended_departments": dept_text,
            "sales_score": 92,
            "inventory_score": 88,
            "delivery_score": 94,
            "cost_score": 58,
            "risk_score": 64,
        },
    ]
    df = pd.DataFrame(base_rows)
    total_weight = max(sum(weights.values()), 1)
    df["ai_score"] = (
        df["sales_score"] * weights["sales"]
        + df["inventory_score"] * weights["inventory"]
        + df["delivery_score"] * weights["delivery"]
        + df["cost_score"] * weights["cost"]
        + df["risk_score"] * weights["risk"]
    ) / total_weight
    budget_bonus = {
        "低": {"保守方案": 6, "均衡方案": 2, "激进方案": -4},
        "中": {"保守方案": 2, "均衡方案": 6, "激进方案": 1},
        "高": {"保守方案": -1, "均衡方案": 3, "激进方案": 6},
    }
    risk_bonus = {
        "保守": {"保守方案": 7, "均衡方案": 3, "激进方案": -5},
        "均衡": {"保守方案": 2, "均衡方案": 7, "激进方案": 2},
        "激进": {"保守方案": -2, "均衡方案": 3, "激进方案": 8},
    }
    cycle_bonus = {
        "7天应急处理": {"保守方案": 5, "均衡方案": 2, "激进方案": -2},
        "15天专项优化": {"保守方案": 2, "均衡方案": 5, "激进方案": 2},
        "30天经营提升": {"保守方案": 0, "均衡方案": 3, "激进方案": 5},
    }
    df["ai_score"] += df["plan_name"].map(budget_bonus.get(budget_level, {})).fillna(0)
    df["ai_score"] += df["plan_name"].map(risk_bonus.get(risk_preference, {})).fillna(0)
    df["ai_score"] += df["plan_name"].map(cycle_bonus.get(cycle, {})).fillna(0)
    df["ai_score"] = df["ai_score"].round(1)
    df["data_basis"] = basis
    df["risk_warning"] = df["risk_level"].map(
        {
            "低": "需要防止改善幅度不足。",
            "中": "需要跨部门配合，关注执行节奏。",
            "高": "需要预算审批和试点风险控制。",
        }
    )
    df["ai_recommendation"] = df["ai_score"].rank(method="first", ascending=False).astype(int)
    return df.sort_values("ai_score", ascending=False)[
        [
            "ai_recommendation",
            "plan_name",
            "plan_position",
            "core_actions",
            "data_basis",
            "expected_effect",
            "estimated_cost",
            "execution_difficulty",
            "risk_level",
            "risk_warning",
            "recommended_departments",
            "ai_score",
        ]
    ]


def expert_consultation_df(metrics: dict[str, float | int], problem: str, recommended_plan: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "expert_role": "库存专家",
                "focus_area": "商品结构与库存资金",
                "expert_opinion": f"建议保障A类商品，同时控制C类库存占用。当前缺货风险商品{metrics['shortage_products']}种，C类库存金额占比{metrics['c_stock_rate']}%。",
                "support_plan": recommended_plan,
            },
            {
                "expert_role": "配送专家",
                "focus_area": "异常订单与网点履约",
                "expert_opinion": f"建议优先治理高风险网点和重点超时订单。当前异常配送订单{metrics['exception_orders']}条，高风险网点{metrics['high_risk_outlets']}个。",
                "support_plan": recommended_plan,
            },
            {
                "expert_role": "财务专家",
                "focus_area": "投入产出与资金占用",
                "expert_opinion": f"建议先做投入产出更清晰的任务，预计清仓可释放库存资金{metrics['release_amount']}元。",
                "support_plan": recommended_plan,
            },
            {
                "expert_role": "低空运营专家",
                "focus_area": "低空试点可行性",
                "expert_opinion": f"低空配送适合服务A类、生鲜冷链和乡镇异常订单。当前推荐起降点{metrics['launch_points']}个，可低空覆盖订单{metrics['route_covered_orders']}单。",
                "support_plan": recommended_plan,
            },
            {
                "expert_role": "经营决策助手",
                "focus_area": problem,
                "expert_opinion": f"综合库存、配送、财务和低空试点意见，建议优先采用“{recommended_plan}”，再由管理人员确认。",
                "support_plan": recommended_plan,
            },
        ]
    )


def confirmed_plan_tasks(plan: pd.Series, metrics: dict[str, float | int], departments: list[str], cycle: str) -> pd.DataFrame:
    deadline = {"7天应急处理": "7天内", "15天专项优化": "15天内", "30天经营提升": "30天内"}.get(cycle, "30天内")
    all_tasks = {
        "采购部": ("复核A类缺货商品并生成补货清单", f"缺货风险商品{metrics['shortage_products']}种"),
        "仓储部": ("处理C类高库存商品并调整仓位", f"建议清仓商品{metrics['high_overstock_products']}种"),
        "配送部": ("优先处理A类和生鲜冷链超时订单", f"A类超时{metrics['a_timeout_orders']}条，生鲜冷链超时{metrics['fresh_timeout_orders']}条"),
        "网点运营部": ("治理乡镇高风险网点并调整服务能力", f"乡镇高风险网点{metrics['township_high_risk_outlets']}个"),
        "低空项目组": ("按推荐起降点验证低空应急配送航线", f"推荐起降点{metrics['launch_points']}个，可低空覆盖订单{metrics['route_covered_orders']}单"),
        "管理层": ("确认预算、周期和复盘指标", f"采用{plan['plan_name']}，AI评分{plan['ai_score']}"),
    }
    selected = departments or list(all_tasks)
    rows = []
    for department in selected:
        if department not in all_tasks:
            continue
        task, basis = all_tasks[department]
        rows.append(
            {
                "department": department,
                "task": task,
                "deadline": deadline,
                "data_basis": basis,
                "expected_output": "形成执行记录并纳入下次经营复盘",
            }
        )
    return pd.DataFrame(rows)


def confirmed_plan_effects(metrics: dict[str, float | int], plan_name: str, cycle: str) -> pd.DataFrame:
    plan_factor = {"保守方案": 0.55, "均衡方案": 0.78, "激进方案": 1.0}.get(plan_name, 0.78)
    cycle_factor = {"7天应急处理": 0.55, "15天专项优化": 0.78, "30天经营提升": 1.0}.get(cycle, 0.78)
    factor = plan_factor * cycle_factor
    timeout_base = metrics["a_timeout_orders"] + metrics["fresh_timeout_orders"] + metrics["township_timeout_orders"]
    rows = [
        ("缺货风险商品", f"{metrics['shortage_products']}种", int(metrics["shortage_products"] * 0.45 * factor), "预计减少缺货商品数"),
        ("库存资金释放", f"{metrics['release_amount']}元", int(metrics["release_amount"] * 0.38 * factor), "预计释放库存资金"),
        ("重点超时订单", f"{timeout_base}条", int(timeout_base * 0.36 * factor), "预计减少重点超时订单"),
        ("低空覆盖订单", f"{metrics['route_covered_orders']}单", int(metrics["route_covered_orders"] * 0.55 * factor), "预计纳入低空应急覆盖"),
    ]
    return pd.DataFrame(rows, columns=["kpi_name", "current_value", "expected_improvement", "improvement_desc"])


def decision_record_text(
    problem: str,
    budget_level: str,
    cycle: str,
    risk_preference: str,
    plan: pd.Series,
    decision_opinion: str,
) -> str:
    return (
        f"AI经营决策记录\n"
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"经营问题：{problem}\n"
        f"预算水平：{budget_level}\n"
        f"执行周期：{cycle}\n"
        f"风险偏好：{risk_preference}\n"
        f"确认方案：{plan['plan_name']}\n"
        f"AI评分：{plan['ai_score']}\n"
        f"核心措施：{plan['core_actions']}\n"
        f"数据依据：{plan['data_basis']}\n"
        f"风险提示：{plan['risk_warning']}\n"
        f"人工决策意见：{decision_opinion.strip() or '同意采用该方案，并进入部门执行。'}\n"
    )


def generate_report_markdown(metrics: dict[str, float | int], report_type: str, prompt: str) -> str:
    rag_count = len(read_csv(RAG_KB_FILE)) if RAG_KB_FILE.exists() else 0
    rag_context = read_csv(RAG_RETRIEVAL_FILE) if RAG_RETRIEVAL_FILE.exists() else pd.DataFrame()
    rag_sources = "、".join(rag_context["source_file"].dropna().astype(str).head(4).tolist()) if not rag_context.empty and "source_file" in rag_context.columns else "暂未执行RAG召回"
    return f"""# 物流企业智能运营经营诊断报告

报告类型：{report_type}

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、项目背景与数据来源

本报告面向县域中小物流企业经营管理场景，基于数据治理、AI库存分析、配送异常诊断、网点风险评分、K-Means配送片区划分、低空起降点筛选和低空航线规划结果生成。报告目标是把技术分析结果转化为管理层可以直接理解和执行的经营建议。

本次报告综合读取模块一、模块二和模块三的输出结果，包括商品基础指标表、ABC分类结果、销量预测结果、库存预警结果、补货清仓建议、配送异常诊断结果、网点风险评分结果、聚类片区结果、候选起降点结果和低空航线规划结果。同时，系统构建RAG经营知识库，共沉淀{rag_count}条知识片段，最近一次召回来源包括：{rag_sources}。

## 二、企业当前经营核心问题

系统识别出四类核心问题：

1. A类重点商品需要优先保障。当前A类商品{metrics['a_products']}种，贡献约{metrics['a_sales_rate']}%销售额，是企业销售基本盘。
2. C类商品库存占用偏高。当前C类商品{metrics['c_products']}种，销售贡献约{metrics['c_sales_rate']}%，但库存金额占比达到{metrics['c_stock_rate']}%，存在资金占用风险。
3. 重点商品和生鲜冷链配送存在时效风险。系统识别异常配送订单{metrics['exception_orders']}条，其中A类重点商品超时{metrics['a_timeout_orders']}条，生鲜冷链超时{metrics['fresh_timeout_orders']}条，超长配送订单{metrics['long_timeout_orders']}条。
4. 乡镇高风险网点需要重点治理。系统识别高风险网点{metrics['high_risk_outlets']}个，其中乡镇高风险网点{metrics['township_high_risk_outlets']}个。

## 三、库存结构与商品经营分析

ABC分类显示，A类商品数量少但销售贡献高，应作为重点补货和重点配送对象。B类商品经营相对稳定，应维持安全库存。C类商品数量多、销售贡献低，但库存资金占用明显，需要通过清仓、组合销售和停止采购降低占用。

系统识别缺货风险商品{metrics['shortage_products']}种，建议补货金额约{metrics['purchase_amount']}元；建议清仓商品{metrics['high_overstock_products']}种，预计释放库存资金约{metrics['release_amount']}元。

## 四、配送异常与网点风险分析

配送异常主要集中在A类重点商品、生鲜冷链订单和乡镇高风险网点。模块三已把异常订单从订单层面汇总到网点层面，并结合网点基础数据生成风险评分。

当前高风险网点{metrics['high_risk_outlets']}个，中风险网点{metrics['medium_risk_outlets']}个，低风险网点{metrics['low_risk_outlets']}个。风险来源包括平均配送时长偏高、历史超时率较高、道路等级较低、服务能力不足以及高优先级商品超时集中。

## 五、低空应急配送规划分析

K-Means已将{metrics['outlet_total']}个网点划分为{metrics['cluster_count']}个配送优化片区。系统筛选{metrics['launch_candidates']}个候选起降点，推荐{metrics['launch_points']}个低空应急起降点。

低空配送不建议服务所有普通订单，而应优先服务A类重点商品、生鲜冷链商品和乡镇高风险异常订单。当前系统对{metrics['route_planned_orders']}个高优先级异常订单进行航线规划，其中{metrics['route_covered_orders']}个可低空覆盖，{metrics['route_uncovered_orders']}个暂不建议低空配送。

## 六、综合决策建议

1. 对A类缺货商品优先补货，保障核心销售额。
2. 对C类高库存商品执行清仓、组合销售或停止采购，释放库存资金。
3. 对A类重点商品和生鲜冷链超时订单优先调度，必要时纳入低空应急配送。
4. 对乡镇高风险网点进行资源倾斜，结合4个配送片区优化末端调度。
5. 以推荐起降点作为低空应急配送试点，逐步接入真实空域、气象和无人机飞控数据。

## 七、预期价值与实施路径

通过以上措施，企业可以降低重点商品缺货风险，释放C类商品库存资金，提升乡镇网点配送效率，保障生鲜冷链和A类重点商品时效，并为低空经济在县域物流中的应用提供试点依据。

## 附：本次提示词摘要

{prompt.strip()}

## 附：RAG检索增强说明

本报告采用离线RAG模拟流程：先把企业制度文档、ABC分类结果、库存预警结果、配送异常结果、网点风险结果和低空航线结果转化为知识片段，再根据经营问题进行Top-K召回，最后将召回上下文与提示词共同用于报告生成。第一版不联网、不调用真实向量模型，但保留后续接入真实Embedding、向量库和大模型API的接口。
"""


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_lines = []
    in_list = False
    for line in lines:
        text = line.strip()
        if not text:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if text.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h1>{html.escape(text[2:])}</h1>")
        elif text.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{html.escape(text[3:])}</h2>")
        elif text.startswith("- ") or text[:2].isdigit() and ". " in text[:4]:
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item = text[2:] if text.startswith("- ") else text.split(". ", 1)[1]
            html_lines.append(f"<li>{html.escape(item)}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{html.escape(text)}</p>")
    if in_list:
        html_lines.append("</ul>")
    body = "\n".join(html_lines)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>物流企业经营诊断报告</title>
<style>
body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f5f7fb; color: #172033; }}
.report {{ max-width: 1040px; margin: 28px auto; padding: 34px 42px; background: #fff; border: 1px solid #dbe3ef; border-radius: 8px; box-shadow: 0 8px 28px rgba(15, 23, 42, .08); }}
h1 {{ font-size: 30px; margin: 0 0 18px; color: #0f172a; }}
h2 {{ font-size: 21px; margin: 28px 0 10px; color: #124170; border-left: 4px solid #2f80ff; padding-left: 10px; }}
p, li {{ line-height: 1.8; font-size: 15px; }}
ul {{ padding-left: 24px; }}
</style>
</head>
<body><main class="report">{body}</main></body>
</html>"""


def save_report(metrics: dict[str, float | int], report_type: str, prompt: str) -> tuple[Path, Path, str]:
    if not RAG_KB_FILE.exists():
        build_rag_knowledge_base(
            [path.name for path in sample_doc_files()],
            result_file_options(),
            300,
            50,
            metrics,
        )
    if not RAG_RETRIEVAL_FILE.exists():
        retrieve_rag_context(st.session_state.get("module4_question", "企业下一步应该优先做哪些调整？"), 4, 0.65)
    markdown_text = generate_report_markdown(metrics, report_type, prompt)
    html_text = markdown_to_html(markdown_text)
    md_path = write_text(markdown_text, REPORT_DIR / "business_diagnosis_report.md")
    html_path = write_text(html_text, REPORT_DIR / "business_diagnosis_report.html")
    write_log("生成智能经营诊断报告", "多模块结果文件", "business_diagnosis_report.md;business_diagnosis_report.html")
    return md_path, html_path, markdown_text


def presentation_summary_text() -> str:
    return (
        "我负责的是大模型经营决策模块。系统接收前三个模块的结果，包括数据清洗结果、AI库存分析结果、配送风险诊断结果和低空航线规划结果。"
        "然后我通过提示词引导大模型生成经营诊断报告。报告指出，企业当前存在C类商品库存占用过高、A类商品缺货风险、乡镇网点配送超时、"
        "生鲜冷链订单保障不足等问题。系统进一步提出A类商品优先补货、C类商品清仓、乡镇高风险网点优化和低空应急配送试点建议。"
        "最终形成面向企业管理层的综合经营决策报告，为物流企业智能运营提供数据支撑和决策依据。"
    )


def export_all(metrics: dict[str, float | int], report_type: str, prompt: str, question: str = "", answer: str = "") -> list[Path]:
    md_path, html_path, _ = save_report(metrics, report_type, prompt)
    if not RAG_KB_FILE.exists():
        build_rag_knowledge_base([path.name for path in sample_doc_files()], result_file_options(), 300, 50, metrics)
    if not RAG_RETRIEVAL_FILE.exists():
        retrieve_rag_context(question or "企业下一步应该优先做哪些调整？", 4, 0.65)
    summary_path = write_csv(management_summary_df(metrics), REPORT_DIR / "management_summary.csv")
    actions_path = REPORT_DIR / "decision_actions.csv"
    if not actions_path.exists():
        actions_path = write_csv(decision_actions_df(metrics), actions_path)
    presentation_path = write_text(presentation_summary_text(), REPORT_DIR / "presentation_summary.txt")
    prompt_path = save_prompt(prompt, report_type)
    response_text = answer or mock_answer(question or "企业下一步应该优先做哪些调整？", metrics)
    response_path = save_answer_record(question or "企业下一步应该优先做哪些调整？", response_text)
    write_log(
        "导出报告与比赛总结",
        "business_diagnosis_report",
        "management_summary.csv;decision_actions.csv;presentation_summary.txt",
        remark="导出完成",
    )
    output_paths = [md_path, html_path, summary_path, actions_path, presentation_path, prompt_path, response_path]
    output_paths.extend(
        [
            path
            for path in [
                REPORT_DIR / "decision_plan_compare.csv",
                REPORT_DIR / "expert_consultation.csv",
                REPORT_DIR / "department_action_tasks.csv",
                REPORT_DIR / "decision_effect_simulation.csv",
                REPORT_DIR / "decision_record.txt",
            ]
            if path.exists()
        ]
    )
    output_paths.extend([path for path in [RAG_KB_FILE, RAG_RETRIEVAL_FILE] if path.exists()])
    return output_paths


def page_receive() -> None:
    st.header("多模块结果接收与校验")
    st.caption("接收模块一、模块二、模块三结果文件，形成经营报告输入数据包。")

    status = file_status()
    received = int((status["status"] == "已接收").sum())
    metric_cards(
        [
            ("结果文件", f"{received}/{len(status)}", "模块四输入"),
            ("模块一", f"{int((status['module'].str.contains('模块一') & (status['status'] == '已接收')).sum())}项", "数据治理结果"),
            ("模块二", f"{int((status['module'].str.contains('模块二') & (status['status'] == '已接收')).sum())}项", "库存分析结果"),
            ("模块三", f"{int((status['module'].str.contains('模块三') & (status['status'] == '已接收')).sum())}项", "配送规划结果"),
        ]
    )

    status_preview = status[["module", "display_name", "status"]]
    st.dataframe(status_preview, use_container_width=True, hide_index=True, height=220)

    receive_clicked = st.button("接收前三个模块结果", type="primary", use_container_width=True)

    if receive_clicked:
        render_timed_steps(
            [
                "正在接收模块一数据治理结果...",
                "正在接收模块二库存分析与预测结果...",
                "正在接收模块三配送规划与低空航线结果...",
                "正在汇总核心经营指标...",
            ],
            duration_seconds=3.0,
        )
        ensure_demo_outputs()
        status = file_status()
        metrics = build_metrics()
        st.session_state["module4_metrics"] = metrics
        st.session_state["module4_received"] = True
        st.session_state["module4_rag_ready"] = False
        st.session_state["module4_rag_context"] = pd.DataFrame()
        write_log("接收前三模块结果", f"{len(status)}个文件", "management_summary.csv", remark="核心经营指标已汇总")
        st.success(f"模块一、模块二、模块三结果接收完成，共读取 {len(status)} 个结果文件，核心经营指标已汇总。")
        scroll_to_page_bottom()

    if not st.session_state.get("module4_received"):
        st.info("请先点击“接收前三个模块结果”，再构建RAG知识库和查看核心经营指标。")
        return

    metrics = st.session_state.get("module4_metrics", build_metrics())

    st.header("RAG经营知识库接入")
    st.caption("把企业制度文档和前三个模块结果转化为可召回知识，用于后续经营问答与报告生成。")
    doc_options = [path.name for path in sample_doc_files()]
    result_options = result_file_options()
    rag_left, rag_right = st.columns([1.1, 1])
    with rag_left:
        selected_docs = st.multiselect("选择企业制度知识文档", doc_options, default=doc_options)
        selected_results = st.multiselect(
            "选择进入RAG知识库的模块结果",
            result_options,
            default=[],
        )
    with rag_right:
        chunk_size = st.slider("Chunk大小/字", 0, 600, 0, step=20)
        chunk_overlap = st.slider("Chunk重叠/字", 0, 120, 0, step=10)
        top_k = st.slider("Top-K召回数量", 0, 8, 0)
        similarity_threshold = st.slider("相似度阈值", 0.00, 0.90, 0.00, step=0.05)
        retrieval_mode = st.selectbox("检索方式", ["混合检索", "关键词检索", "语义检索模拟"], index=0)

    build_rag_clicked = st.button("构建RAG经营知识库", type="primary", use_container_width=True)

    if build_rag_clicked:
        render_timed_steps(
            [
                "正在读取企业制度文档...",
                "正在读取已选择的模块结果文件...",
                "正在按Chunk参数切分知识片段...",
                "正在提取经营关键词和业务标签...",
                "正在写入RAG知识库文件...",
                "正在生成知识库预览结果...",
            ],
            duration_seconds=6.0,
        )
        knowledge = build_rag_knowledge_base(selected_docs, selected_results, chunk_size, chunk_overlap, metrics)
        st.session_state["module4_rag_ready"] = True
        st.session_state["module4_rag_config"] = {
            "top_k": top_k,
            "threshold": similarity_threshold,
            "retrieval_mode": retrieval_mode,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        st.success(f"RAG经营知识库构建完成：共生成 {len(knowledge)} 条知识片段。")
        st.dataframe(knowledge, use_container_width=True, hide_index=True, height=220)
        scroll_to_page_bottom()

    if st.session_state.get("module4_rag_ready") and RAG_KB_FILE.exists():
        knowledge = read_csv(RAG_KB_FILE)
        metric_cards(
            [
                ("RAG知识片段", f"{len(knowledge)}条", "rag_knowledge_base.csv"),
                ("制度文档", f"{len(doc_options)}个", "data/sample_docs"),
                ("检索Top-K", str(st.session_state.get("module4_rag_config", {}).get("top_k", top_k)), retrieval_mode),
                ("相似度阈值", f"{st.session_state.get('module4_rag_config', {}).get('threshold', similarity_threshold):.2f}", "召回过滤"),
            ]
        )
        st.subheader("知识库指标摘要")
        metric_cards(
            [
                ("缺货风险商品", f"{metrics['shortage_products']}种", "库存预警"),
                ("建议清仓商品", f"{metrics['high_overstock_products']}种", "库存优化"),
                ("异常配送订单", f"{metrics['exception_orders']}条", "配送诊断"),
                ("高风险网点", f"{metrics['high_risk_outlets']}个", "网点治理"),
                ("可低空覆盖订单", f"{metrics['route_covered_orders']}单", "航线分配"),
            ]
        )

def page_qa() -> None:
    st.header("经营问题选择与智能问答")
    st.caption("通过大模型对话窗口提出经营问题，系统基于前三个模块结果和RAG知识库生成经营回答。")
    metrics = st.session_state.get("module4_metrics", build_metrics())

    def retrieval_file_summary(retrieved: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(retrieved, pd.DataFrame) or retrieved.empty:
            return pd.DataFrame(columns=["召回文件", "知识类别", "最高相似度", "召回片段数"])
        view = retrieved.copy()
        if "source_file" not in view.columns:
            return pd.DataFrame(columns=["召回文件", "知识类别", "最高相似度", "召回片段数"])
        if "knowledge_category" not in view.columns:
            view["knowledge_category"] = "经营知识"
        if "similarity_score" not in view.columns:
            view["similarity_score"] = 0
        view["similarity_score"] = pd.to_numeric(view["similarity_score"], errors="coerce").fillna(0)
        summary = (
            view.groupby(["source_file", "knowledge_category"], as_index=False)
            .agg(similarity_score=("similarity_score", "max"), chunk_count=("source_file", "count"))
            .sort_values("similarity_score", ascending=False)
        )
        summary["similarity_score"] = summary["similarity_score"].round(3)
        return summary.rename(
            columns={
                "source_file": "召回文件",
                "knowledge_category": "知识类别",
                "similarity_score": "最高相似度",
                "chunk_count": "召回片段数",
            }
        )

    def render_retrieval_files_table(rows: list[dict[str, object]] | pd.DataFrame) -> None:
        summary = pd.DataFrame(rows) if isinstance(rows, list) else rows
        if summary.empty:
            return
        table_html = summary.to_html(index=False, border=0, classes="qa-rag-file-table", escape=True)
        st.markdown(f'<div class="qa-rag-file-wrap">{table_html}</div>', unsafe_allow_html=True)

    if "module4_qa_messages" not in st.session_state:
        st.session_state["module4_qa_messages"] = [
            {
                "role": "assistant",
                "content": "你好，我是经营决策助手。请围绕库存、配送、网点风险、低空应急配送或经营报告提出问题。",
            }
        ]

    def run_qa_dialog(question: str) -> None:
        render_timed_steps(
            [
                "正在读取ABC分类结果...",
                "正在读取库存预警结果...",
                "正在检索RAG经营知识库...",
                "正在读取配送异常结果...",
                "正在读取低空航线规划结果...",
                "正在生成经营回答...",
            ],
            duration_seconds=3.0,
        )
        rag_config = st.session_state.get("module4_rag_config", {"top_k": 4, "threshold": 0.65})
        if not RAG_KB_FILE.exists():
            build_rag_knowledge_base(
                [path.name for path in sample_doc_files()],
                result_file_options(),
                300,
                50,
                metrics,
            )
            st.session_state["module4_rag_ready"] = True
        top_k_value = int(rag_config.get("top_k", 4)) or 4
        threshold_value = float(rag_config.get("threshold", 0.65))
        retrieved = retrieve_rag_context(question, top_k_value, threshold_value)
        st.session_state["module4_rag_context"] = retrieved
        st.session_state["module4_has_asked_question"] = True
        answer = mock_answer(question, metrics)
        st.session_state["module4_question"] = question
        st.session_state["module4_answer"] = answer
        st.session_state["module4_qa_messages"].append({"role": "assistant", "content": answer})
        write_log("生成经营问答回答", question, "llm_response_mock.txt", remark="页面预览")

    st.markdown("#### 经营决策对话")
    for message in st.session_state["module4_qa_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    rag_context = st.session_state.get("module4_rag_context")
    has_asked_question = st.session_state.get("module4_has_asked_question", False)
    if has_asked_question and (not isinstance(rag_context, pd.DataFrame) or rag_context.empty) and RAG_RETRIEVAL_FILE.exists():
        rag_context = read_csv(RAG_RETRIEVAL_FILE)
        st.session_state["module4_rag_context"] = rag_context
    if has_asked_question and isinstance(rag_context, pd.DataFrame) and not rag_context.empty:
        st.markdown("#### 最近一次RAG知识召回结果")
        st.caption("用于展示当前问题实际召回的知识片段和来源文件。")
        render_retrieval_files_table(retrieval_file_summary(rag_context))
        st.dataframe(rag_context, use_container_width=True, hide_index=True, height=220)

    chat_question = st.chat_input("向经营决策助手提问，例如：企业下一步应该优先做哪些调整？", key="module4_qa_chat_input")
    if chat_question:
        st.session_state["module4_qa_messages"].append({"role": "user", "content": chat_question})
        run_qa_dialog(chat_question)
        st.rerun()


def page_report_center() -> None:
    st.header("大模型报告编写与生成")
    st.caption("在同一页面完成提示词编写、大模型分析过程和经营诊断报告生成。")
    metrics = st.session_state.get("module4_metrics", build_metrics())

    if "module4_prompt_text" not in st.session_state:
        st.session_state["module4_prompt_text"] = ""
    if "module4_simple_report_requirement" not in st.session_state:
        st.session_state["module4_simple_report_requirement"] = ""

    st.subheader("提示词编写")
    report_type = st.selectbox("报告类型", REPORT_TYPES, index=0)
    simple_requirement = st.text_area("简单报告需求", height=95, key="module4_simple_report_requirement")
    prompt_btn_left, prompt_btn_right = st.columns(2)
    with prompt_btn_left:
        optimize_clicked = st.button("提示词优化", type="primary", use_container_width=True)
    with prompt_btn_right:
        template_clicked = st.button("加载提示词模板", use_container_width=True)

    if optimize_clicked:
        render_timed_steps(
            [
                "正在理解简单报告需求...",
                "正在匹配报告类型和数据来源...",
                "正在组织管理层汇报口径...",
                "正在生成优化后的提示词...",
            ],
            duration_seconds=3.0,
        )
        st.session_state["module4_prompt_text"] = optimize_report_prompt(simple_requirement, report_type)
        write_log("优化报告提示词", simple_requirement, "llm_prompt_record.txt", remark=report_type)
        st.success("已根据简单需求生成优化后的报告提示词。")
        scroll_to_page_bottom()

    if template_clicked:
        st.session_state["module4_prompt_text"] = DEFAULT_PROMPT

    prompt = st.text_area("报告生成提示词", height=260, key="module4_prompt_text")

    prompt_col, analysis_col = st.columns(2)
    with prompt_col:
        save_prompt_clicked = st.button("保存提示词", use_container_width=True)
    with analysis_col:
        analysis_clicked = st.button("开始大模型分析", type="primary", use_container_width=True)

    if save_prompt_clicked:
        path = save_prompt(prompt, report_type)
        st.success(f"已保存提示词：{path.relative_to(PROJECT_ROOT)}")

    st.divider()
    st.subheader("大模型分析过程")
    if analysis_clicked:
        save_prompt(prompt, report_type)
        st.session_state["module4_report_type"] = report_type
        st.session_state["module4_prompt_submitted"] = True
        st.session_state.pop("module4_report_markdown", None)
        st.session_state.pop("module4_report_paths", None)
        progress = st.progress(0)
        log_box = st.empty()
        lines = []
        for idx, step in enumerate(ANALYSIS_STEPS, start=1):
            lines.append(step)
            log_box.code("\n".join(lines), language="text")
            progress.progress(idx / len(ANALYSIS_STEPS))
            time.sleep(6.0 / len(ANALYSIS_STEPS))
        st.session_state["module4_analysis_done"] = True
        st.session_state["module4_analysis_log"] = lines
        write_log("执行大模型分析过程", "多模块结果文件", "report_outline", remark="生成报告大纲")
        st.success("分析完成，已生成报告大纲。")

    if st.session_state.get("module4_analysis_done"):
        analysis_log = st.session_state.get("module4_analysis_log", ANALYSIS_STEPS)
        with st.expander("查看分析过程日志", expanded=False):
            st.code("\n".join(analysis_log), language="text")
        outline_df = pd.DataFrame([(idx + 1, item) for idx, item in enumerate(REPORT_OUTLINE)], columns=["序号", "报告章节"])
        st.dataframe(outline_df, use_container_width=True, hide_index=True, height=250)
    else:
        st.info("请先完成提示词编写，然后点击“开始大模型分析”。")

    st.divider()
    st.subheader("智能经营诊断报告")
    report_col, dashboard_col = st.columns([1, 1])
    with report_col:
        generate_clicked = st.button(
            "生成经营诊断报告",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.get("module4_analysis_done"),
        )
    with dashboard_col:
        dashboard_clicked = st.button("进入决策看板", use_container_width=True)

    if generate_clicked:
        report_type = st.session_state.get("module4_report_type", report_type)
        with st.spinner("正在生成经营诊断报告..."):
            time.sleep(6.0)
        md_path, html_path, markdown_text = save_report(metrics, report_type, prompt)
        stream_markdown_output(markdown_text, duration_seconds=10.0)
        st.session_state["module4_report_markdown"] = markdown_text
        st.session_state["module4_report_paths"] = (str(md_path), str(html_path))
        st.success(f"已生成报告：{md_path.name}、{html_path.name}")
        scroll_to_page_bottom()

    markdown_text = st.session_state.get("module4_report_markdown", "")
    if markdown_text:
        sections = markdown_text.split("\n## ")
        title = sections[0]
        content_sections = ["## " + item for item in sections[1:]]
        preview_left, preview_right = st.columns([0.8, 1.5])
        with preview_left:
            st.markdown("#### 报告目录")
            for item in REPORT_OUTLINE:
                st.markdown(f"- {item}")
            st.markdown("#### 关键结论")
            st.write(f"A类商品 {metrics['a_products']} 种，贡献 {metrics['a_sales_rate']}% 销售额")
            st.write(f"C类商品 {metrics['c_products']} 种，库存金额占比 {metrics['c_stock_rate']}%")
            st.write(f"异常配送订单 {metrics['exception_orders']} 条")
            st.write(f"候选起降点 {metrics['launch_candidates']} 个，推荐起降点 {metrics['launch_points']} 个")
        with preview_right:
            st.markdown(title)
            for section in content_sections:
                heading = section.splitlines()[0].replace("## ", "")
                with st.expander(heading, expanded=heading.startswith("一、") or heading.startswith("二、")):
                    st.markdown(section)
    else:
        st.info("完成大模型分析后，点击“生成经营诊断报告”查看报告预览。")

    if dashboard_clicked:
        set_next_page("AI智能决策驾驶舱")
        st.rerun()


def page_prompt() -> None:
    st.header("大模型报告提示词编写")
    st.caption("选择报告类型并编写提示词，明确数据来源、关键指标、报告结构和读者对象。")
    metrics = st.session_state.get("module4_metrics", build_metrics())
    status = file_status()

    left, right = st.columns([0.9, 1.3])
    with left:
        st.subheader("可用数据文件")
        st.dataframe(status[["module", "file_name", "status"]], use_container_width=True, hide_index=True, height=220)
        st.subheader("RAG知识库状态")
        if RAG_KB_FILE.exists():
            knowledge = read_csv(RAG_KB_FILE)
            st.success(f"已构建RAG知识库：{len(knowledge)}条知识片段")
            st.dataframe(knowledge[["knowledge_id", "source_type", "source_file", "knowledge_category"]].head(8), use_container_width=True, hide_index=True)
        else:
            st.info("请先在第一页接收前三个模块结果。")
        rag_context = st.session_state.get("module4_rag_context")
        if isinstance(rag_context, pd.DataFrame) and not rag_context.empty:
            st.subheader("最近一次RAG召回上下文")
            st.dataframe(rag_context.head(4), use_container_width=True, hide_index=True)
        st.subheader("可引用核心指标")
        st.dataframe(management_summary_df(metrics).head(12), use_container_width=True, hide_index=True)
        st.subheader("报告结构模板")
        st.write("\n".join(REPORT_OUTLINE))

    with right:
        report_type = st.selectbox("报告类型", REPORT_TYPES, index=0)
        if st.button("加载提示词模板"):
            st.session_state["module4_prompt_text"] = DEFAULT_PROMPT
        if "module4_prompt_text" not in st.session_state:
            st.session_state["module4_prompt_text"] = DEFAULT_PROMPT
        prompt = st.text_area("报告生成提示词", height=360, key="module4_prompt_text")

        col1, col2 = st.columns(2)
        with col1:
            save_clicked = st.button("保存提示词", use_container_width=True)
        with col2:
            submit_clicked = st.button("提交给大模型生成报告", type="primary", use_container_width=True)

    if save_clicked:
        path = save_prompt(prompt, report_type)
        st.success(f"已保存提示词：{path.relative_to(PROJECT_ROOT)}")

    if submit_clicked:
        path = save_prompt(prompt, report_type)
        st.session_state["module4_report_type"] = report_type
        st.session_state["module4_prompt_submitted"] = True
        write_log("提交报告提示词", "llm_prompt_record.txt", "business_diagnosis_report.md", remark="进入模拟分析")
        st.success(f"提示词已提交：{path.relative_to(PROJECT_ROOT)}")
        set_next_page("大模型模拟分析过程")
        st.rerun()


def page_analysis() -> None:
    st.header("大模型模拟分析过程")
    st.caption("展示大模型读取多模块结果、形成报告大纲和组织决策建议的过程。")

    if not st.session_state.get("module4_prompt_submitted"):
        st.info("请先在“大模型报告提示词编写”页面提交报告提示词。")

    if st.button("开始模拟分析", type="primary"):
        progress = st.progress(0)
        log_box = st.empty()
        lines = []
        for idx, step in enumerate(ANALYSIS_STEPS, start=1):
            lines.append(step)
            log_box.code("\n".join(lines), language="text")
            progress.progress(idx / len(ANALYSIS_STEPS))
            time.sleep(0.1)
        st.session_state["module4_analysis_done"] = True
        write_log("执行大模型模拟分析过程", "多模块结果文件", "report_outline", remark="生成报告大纲")
        st.success("分析过程完成，已生成报告大纲。")

    if st.session_state.get("module4_analysis_done"):
        st.subheader("报告大纲")
        outline_df = pd.DataFrame([(idx + 1, item) for idx, item in enumerate(REPORT_OUTLINE)], columns=["序号", "章节"])
        st.dataframe(outline_df, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认生成完整报告", type="primary", use_container_width=True):
                set_next_page("智能经营诊断报告生成")
                st.rerun()
        with col2:
            if st.button("返回修改提示词", use_container_width=True):
                set_next_page("大模型报告提示词编写")
                st.rerun()


def page_report() -> None:
    st.header("智能经营诊断报告生成")
    st.caption("生成面向企业管理层的经营诊断报告，并保存 Markdown 与 HTML 版本。")
    metrics = st.session_state.get("module4_metrics", build_metrics())
    prompt = get_report_prompt()
    report_type = st.session_state.get("module4_report_type", "综合经营诊断报告")

    col1, col2, col3 = st.columns(3)
    with col1:
        generate_clicked = st.button("生成完整报告", type="primary", use_container_width=True)
    with col2:
        regenerate_clicked = st.button("重新生成报告", use_container_width=True)
    with col3:
        dashboard_clicked = st.button("进入决策看板", use_container_width=True)

    if generate_clicked or regenerate_clicked:
        md_path, html_path, markdown_text = save_report(metrics, report_type, prompt)
        st.session_state["module4_report_markdown"] = markdown_text
        st.session_state["module4_report_paths"] = (str(md_path), str(html_path))
        st.success(f"已生成报告：{md_path.name}、{html_path.name}")

    markdown_text = st.session_state.get("module4_report_markdown", "")
    if not markdown_text:
        st.info("请点击“生成完整报告”生成经营诊断报告。")
        if dashboard_clicked:
            set_next_page("AI智能决策驾驶舱")
            st.rerun()
        return
    sections = markdown_text.split("\n## ")
    title = sections[0]
    content_sections = ["## " + item for item in sections[1:]]

    left, right = st.columns([0.8, 1.5])
    with left:
        st.subheader("报告目录")
        for item in REPORT_OUTLINE:
            st.markdown(f"- {item}")
        st.subheader("关键结论")
        st.write(f"A类商品 {metrics['a_products']} 种，贡献 {metrics['a_sales_rate']}% 销售额")
        st.write(f"C类商品 {metrics['c_products']} 种，库存金额占比 {metrics['c_stock_rate']}%")
        st.write(f"异常配送订单 {metrics['exception_orders']} 条")
        st.write(f"候选起降点 {metrics['launch_candidates']} 个，推荐起降点 {metrics['launch_points']} 个")
    with right:
        st.markdown(title)
        for section in content_sections:
            heading = section.splitlines()[0].replace("## ", "")
            with st.expander(heading, expanded=heading.startswith("一、") or heading.startswith("二、")):
                st.markdown(section)

    if dashboard_clicked:
        set_next_page("AI智能决策驾驶舱")
        st.rerun()


def page_dashboard() -> None:
    st.header("AI经营决策会商与行动落地")
    st.caption("通过经营问题选择、决策约束设置、大模型方案生成、专家会商和人工确认，形成可执行的部门行动清单。")
    metrics = st.session_state.get("module4_metrics", build_metrics())

    st.subheader("经营关键指标")
    metric_cards(
        [
            ("A类商品", f"{metrics['a_products']}种", "核心销售"),
            ("缺货风险商品", f"{metrics['shortage_products']}种", "补货关注"),
            ("建议清仓商品", f"{metrics['high_overstock_products']}种", "库存释放"),
            ("异常配送订单", f"{metrics['exception_orders']}条", "履约风险"),
            ("高风险网点", f"{metrics['high_risk_outlets']}个", "末端治理"),
        ]
    )

    st.subheader("1. 选择经营问题")
    problem = st.selectbox(
        "本次决策要解决的问题",
        ["库存资金占用过高", "A类商品缺货风险", "乡镇配送超时严重", "生鲜冷链订单履约风险", "低空应急配送试点评估"],
        index=0,
    )
    st.info(problem_data_basis(metrics, problem))

    st.subheader("2. 设置决策约束")
    config_left, config_right = st.columns([1.05, 1])
    with config_left:
        budget_level = st.radio("可接受预算", ["低", "中", "高"], index=0, horizontal=True)
        action_cycle = st.radio("行动周期", ["7天应急处理", "15天专项优化", "30天经营提升"], horizontal=True)
        risk_preference = st.radio("风险偏好", ["保守", "均衡", "激进"], index=0, horizontal=True)
        departments = ["采购部", "仓储部", "配送部", "网点运营部", "低空项目组", "管理层"]
        selected_departments = st.multiselect("重点参与部门", departments, default=[])
    with config_right:
        sales_weight = st.slider("销售保障权重", 0, 40, 0)
        inventory_weight = st.slider("库存资金权重", 0, 40, 0)
        delivery_weight = st.slider("配送时效权重", 0, 40, 0)
        cost_weight = st.slider("成本控制权重", 0, 40, 0)
        risk_weight = st.slider("风险控制权重", 0, 40, 0)

    weights = {
        "sales": sales_weight,
        "inventory": inventory_weight,
        "delivery": delivery_weight,
        "cost": cost_weight,
        "risk": risk_weight,
    }
    st.caption(f"当前权重合计：{sum(weights.values())}")
    generate_clicked = st.button("3. AI生成三套经营方案", type="primary", use_container_width=True)

    if generate_clicked:
        steps = [
            "大模型决策助手正在读取经营诊断报告...",
            "正在检索RAG知识库和前三个模块结果...",
            "正在识别经营问题、预算、周期和风险偏好...",
            "正在生成保守、均衡、激进三套方案...",
            "正在组织库存、配送、财务和低空专家会商...",
            "正在计算方案推荐评分和风险提示...",
        ]
        render_timed_steps(steps, duration_seconds=10.0)
        plans = generate_decision_plans(metrics, problem, budget_level, action_cycle, risk_preference, selected_departments, weights)
        recommended_plan = plans.iloc[0]["plan_name"] if not plans.empty else "均衡方案"
        experts = expert_consultation_df(metrics, problem, recommended_plan)
        st.session_state["module4_decision_plans"] = plans
        st.session_state["module4_expert_opinions"] = experts
        st.session_state["module4_decision_problem"] = problem
        st.session_state["module4_budget_level"] = budget_level
        st.session_state["module4_risk_preference"] = risk_preference
        st.session_state["module4_ai_action_cycle"] = action_cycle
        st.session_state["module4_selected_departments"] = selected_departments
        st.session_state.pop("module4_confirmed_plan", None)
        st.session_state.pop("module4_ai_tasks", None)
        st.session_state.pop("module4_ai_effects", None)
        st.session_state.pop("module4_decision_record", None)
        write_csv(plans, REPORT_DIR / "decision_plan_compare.csv")
        write_csv(experts, REPORT_DIR / "expert_consultation.csv")
        write_log("AI生成三套经营方案", problem, "decision_plan_compare.csv;expert_consultation.csv", remark=action_cycle)
        st.success("AI已生成三套经营方案和专家会商意见。")
        scroll_to_page_bottom()

    plans = st.session_state.get("module4_decision_plans")
    experts = st.session_state.get("module4_expert_opinions")
    if isinstance(plans, pd.DataFrame) and not plans.empty:
        st.subheader("4. 查看三套经营方案")
        render_plan_cards(plans)
        top_plan = plans.iloc[0]
        st.info(
            f"AI推荐优先采用“{top_plan['plan_name']}”。"
            f"推荐依据：{top_plan['data_basis']}；风险提示：{top_plan['risk_warning']}"
        )

        if isinstance(experts, pd.DataFrame) and not experts.empty:
            st.subheader("5. 多专家AI会商")
            expert_tabs = st.tabs(experts["expert_role"].tolist())
            for tab, (_, row) in zip(expert_tabs, experts.iterrows()):
                with tab:
                    st.markdown(f"**关注方向：** {row['focus_area']}")
                    st.write(row["expert_opinion"])
                    st.caption(f"支持方案：{row['support_plan']}")

        st.subheader("6. 人工确认最终方案")
        fixed_plan_order = ["保守方案", "均衡方案", "激进方案"]
        option_names = [name for name in fixed_plan_order if name in set(plans["plan_name"])]
        selected_plan_name = st.radio("选择最终采用方案", option_names, index=0, horizontal=True)
        decision_opinion = st.text_area("填写决策意见", value="", height=90)
        confirm_clicked = st.button("确认采用该方案并生成行动清单", type="primary", use_container_width=True)

        if confirm_clicked:
            selected_plan = plans[plans["plan_name"] == selected_plan_name].iloc[0]
            saved_departments = st.session_state.get("module4_selected_departments", selected_departments)
            saved_cycle = st.session_state.get("module4_ai_action_cycle", action_cycle)
            saved_problem = st.session_state.get("module4_decision_problem", problem)
            saved_budget = st.session_state.get("module4_budget_level", budget_level)
            saved_risk = st.session_state.get("module4_risk_preference", risk_preference)
            tasks = confirmed_plan_tasks(selected_plan, metrics, saved_departments, saved_cycle)
            effects = confirmed_plan_effects(metrics, selected_plan_name, saved_cycle)
            record = decision_record_text(saved_problem, saved_budget, saved_cycle, saved_risk, selected_plan, decision_opinion)
            record_path = write_text(record, REPORT_DIR / "decision_record.txt")
            st.session_state["module4_confirmed_plan"] = selected_plan.to_dict()
            st.session_state["module4_ai_tasks"] = tasks
            st.session_state["module4_ai_effects"] = effects
            st.session_state["module4_decision_record"] = record
            write_csv(tasks, REPORT_DIR / "decision_actions.csv")
            write_csv(tasks, REPORT_DIR / "department_action_tasks.csv")
            write_csv(effects, REPORT_DIR / "decision_effect_simulation.csv")
            write_csv(management_summary_df(metrics), REPORT_DIR / "management_summary.csv")
            write_log("确认AI经营决策方案", selected_plan_name, "decision_record.txt;department_action_tasks.csv", remark=saved_problem)
            st.success(f"已确认采用“{selected_plan_name}”，并生成决策记录：{record_path.relative_to(PROJECT_ROOT)}")

    tasks = st.session_state.get("module4_ai_tasks")
    effects = st.session_state.get("module4_ai_effects")
    record = st.session_state.get("module4_decision_record")
    if isinstance(effects, pd.DataFrame) and not effects.empty:
        st.subheader("预期改善指标")
        render_effect_cards(effects)
    if isinstance(tasks, pd.DataFrame) and not tasks.empty:
        st.subheader("部门行动清单")
        st.dataframe(tasks, use_container_width=True, hide_index=True, height=260)
    if isinstance(record, str) and record:
        with st.expander("查看AI辅助决策记录", expanded=False):
            st.code(record, language="text")

    if not isinstance(plans, pd.DataFrame) or plans.empty:
        st.info("请先选择经营问题、设置约束，然后点击“AI生成三套经营方案”。")


def load_css() -> None:
    px.defaults.template = "plotly_dark"
    px.defaults.color_discrete_sequence = ["#4f8cff", "#42c6a4", "#f4b740", "#7c8cff", "#3dd5f3", "#ff6b7a"]
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    module_css_path = ASSETS_DIR / "module4.css"
    if module_css_path.exists():
        st.markdown(f"<style>{module_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_shell() -> str:
    if "module4_page" not in st.session_state or st.session_state["module4_page"] not in PAGES:
        st.session_state["module4_page"] = PAGES[0]
    next_page = st.session_state.pop("module4_next_page", None)
    if next_page in PAGES:
        st.session_state["module4_page"] = next_page

    with st.sidebar:
        st.markdown("### 物流企业AI智能运营决策平台")
        st.caption("模块四 / 学生4：大模型经营决策工程师")
        selected = st.radio("功能菜单", PAGES, key="module4_page", label_visibility="collapsed")
        st.divider()
        st.markdown("**当前模块**")
        st.write(MODULE_TITLE)
        st.caption("负责经营问答、提示词编写、智能报告生成、决策看板和报告导出。")

    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <span class="topbar-label">Logistics AI Operations / Module 4</span>
                <strong>{APP_TITLE}</strong>
            </div>
            <span class="role-chip">{OPERATOR}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <section class="decision-hero">
            <p class="eyebrow">大模型经营决策与智能报告生成系统</p>
            <h1>{PAGE_TITLE}</h1>
            <p>接收数据治理、库存分析、配送规划与低空航线结果，通过提示词、模拟分析、报告生成和决策驾驶舱，把技术结果转化为管理层可执行建议。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    return selected


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    ensure_dirs()
    load_css()
    selected = render_shell()

    if selected == "多模块结果接收与校验":
        page_receive()
    elif selected == "经营问题选择与智能问答":
        page_qa()
    elif selected == "大模型报告编写与生成":
        page_report_center()
    elif selected == "AI智能决策驾驶舱":
        page_dashboard()


if __name__ == "__main__":
    main()
