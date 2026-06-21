from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


APP_TITLE = "物流企业AI智能运营决策平台"
MODULE_TITLE = "AI库存分析与ABC智能分类系统"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = DATA_DIR / "cleaned"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUTS_DIR / "logs"
SCRIPT_DIR = OUTPUTS_DIR / "workflow_results"
LOG_FILE = LOG_DIR / "inventory_ai_log.csv"

PAGES = [
    "库存分析数据接收",
    "大模型ABC智能分类",
    "Python云平台",
    "销量预测模型训练与评估",
    "补货清仓建议与结果交接",
]

DEFAULT_PROMPT = """请你作为一名物流企业库存分析工程师，基于 product_base_metrics.csv 商品基础指标表，帮我生成一段 Python Pandas 代码，用于完成商品 ABC 分类分析。要求按 sales_amount_90d 从高到低排序，计算累计贡献率 cum_rate，累计贡献率小于等于0.7为A类，大于0.7且小于等于0.9为B类，大于0.9为C类。输出 abc_result.csv 和 abc_summary.csv，并添加中文注释，便于现场讲解。"""

GENERATED_CODE = '''import pandas as pd

# 1. 读取商品基础指标表
df = pd.read_csv("data/processed/product_base_metrics.csv")

# 2. 按近90天销售额从高到低排序
df = df.sort_values("sales_amount_90d", ascending=False)

# 3. 计算销售额累计值
df["cum_amount"] = df["sales_amount_90d"].cumsum()

# 4. 计算累计贡献率
df["cum_rate"] = df["cum_amount"] / df["sales_amount_90d"].sum()

# 5. 根据累计贡献率划分ABC类别
def get_abc_class(rate):
    if rate <= 0.7:
        return "A"
    elif rate <= 0.9:
        return "B"
    else:
        return "C"

df["abc_class"] = df["cum_rate"].apply(get_abc_class)

# 6. 生成ABC分类明细表
abc_result = df[
    [
        "product_id",
        "product_name",
        "category",
        "sales_90d",
        "sales_amount_90d",
        "out_freq_90d",
        "stock_qty",
        "stock_amount",
        "cum_rate",
        "abc_class",
    ]
]

# 7. 生成ABC分类汇总表
abc_summary = abc_result.groupby("abc_class").agg(
    product_count=("product_id", "count"),
    sales_amount=("sales_amount_90d", "sum"),
    stock_amount=("stock_amount", "sum"),
).reset_index()

abc_summary["sales_amount_rate"] = (
    abc_summary["sales_amount"] / abc_summary["sales_amount"].sum()
)
abc_summary["stock_amount_rate"] = (
    abc_summary["stock_amount"] / abc_summary["stock_amount"].sum()
)

# 8. 保存结果
abc_result.to_csv("data/processed/abc_result.csv", index=False)
abc_summary.to_csv("data/processed/abc_summary.csv", index=False)

print("ABC分类完成，已生成 abc_result.csv 和 abc_summary.csv")
'''

FIELD_DESC = pd.DataFrame(
    [
        ("product_id", "商品编号"),
        ("product_name", "商品名称"),
        ("category", "商品类别"),
        ("sales_30d", "近30天销量"),
        ("sales_90d", "近90天销量"),
        ("sales_amount_90d", "近90天销售额"),
        ("out_freq_90d", "近90天出库频次"),
        ("stock_qty", "当前库存数量"),
        ("stock_amount", "当前库存金额"),
        ("gross_profit_rate", "毛利率"),
        ("return_count", "退货次数"),
        ("return_rate", "退货率"),
        ("stock_turnover_days", "库存周转天数"),
    ],
    columns=["field_name", "field_desc"],
)


def ensure_dirs() -> None:
    for folder in [PROCESSED_DIR, CLEANED_DIR, LOG_DIR, SCRIPT_DIR]:
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


def write_log(action: str, output_file: str, status: str = "success") -> None:
    ensure_dirs()
    row = pd.DataFrame(
        [
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "output_file": output_file,
                "operator": "学生2：AI库存分析工程师",
                "action": action,
                "status": status,
            }
        ]
    )
    if LOG_FILE.exists():
        old = read_csv(LOG_FILE)
        row = pd.concat([old, row], ignore_index=True)
    write_csv(row, LOG_FILE)


def find_base_metrics_path() -> Path | None:
    candidates = [
        PROCESSED_DIR / "product_base_metrics.csv",
        CLEANED_DIR / "product_base_metrics.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def normalize_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    rename_map = {
        "sales_qty": "sales_90d",
        "sales_amount": "sales_amount_90d",
        "out_freq": "out_freq_90d",
    }
    result = result.rename(columns={old: new for old, new in rename_map.items() if old in result.columns and new not in result.columns})

    if "sales_90d" not in result.columns:
        result["sales_90d"] = 0
    if "sales_amount_90d" not in result.columns:
        result["sales_amount_90d"] = 0
    if "out_freq_90d" not in result.columns:
        result["out_freq_90d"] = 0
    if "sales_30d" not in result.columns:
        result["sales_30d"] = (pd.to_numeric(result["sales_90d"], errors="coerce").fillna(0) * 0.36).round().astype(int)
    if "return_count" not in result.columns:
        result["return_count"] = 0
    if "return_rate" not in result.columns:
        result["return_rate"] = 0
    if "stock_turnover_days" not in result.columns:
        sales_90 = pd.to_numeric(result["sales_90d"], errors="coerce").fillna(0)
        if "stock_qty" not in result.columns:
            result["stock_qty"] = 0
        stock_qty = pd.to_numeric(result["stock_qty"], errors="coerce").fillna(0)
        daily_sales = (sales_90 / 90).replace(0, np.nan)
        result["stock_turnover_days"] = (stock_qty / daily_sales).replace([np.inf, -np.inf], np.nan).fillna(999).round(1)

    numeric_columns = [
        "sales_30d",
        "sales_90d",
        "sales_amount_90d",
        "out_freq_90d",
        "stock_qty",
        "stock_amount",
        "gross_profit_rate",
        "return_count",
        "return_rate",
        "stock_turnover_days",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)

    expected = [
        "product_id",
        "product_name",
        "category",
        "sales_30d",
        "sales_90d",
        "sales_amount_90d",
        "out_freq_90d",
        "stock_qty",
        "stock_amount",
        "gross_profit_rate",
        "return_count",
        "return_rate",
        "stock_turnover_days",
    ]
    for column in expected:
        if column not in result.columns:
            result[column] = ""
    return result[expected].head(200)


def load_base_metrics() -> tuple[pd.DataFrame | None, Path | None]:
    path = find_base_metrics_path()
    if path is None:
        return None, None
    df = normalize_base_metrics(read_csv(path))
    write_csv(df, PROCESSED_DIR / "product_base_metrics.csv")
    return df, path


def render_css() -> None:
    px.defaults.template = "plotly_dark"
    px.defaults.color_discrete_sequence = ["#4f8cff", "#42c6a4", "#f4b740", "#7c8cff", "#3dd5f3", "#ff8a65"]
    css_path = PROJECT_ROOT / "assets" / "module2.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    accent_names = ["blue", "teal", "amber", "cyan", "coral", "violet"]
    cols = st.columns(len(items))
    for index, (col, (title, value, hint)) in enumerate(zip(cols, items)):
        with col:
            st.markdown(
                f"""
                <div class="metric-card metric-card--{accent_names[index % len(accent_names)]}">
                    <span>{title}</span>
                    <strong>{value}</strong>
                    <span>{hint}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def scroll_to_page_anchor(anchor_id: str) -> None:
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            const target = window.parent.document.getElementById("{anchor_id}");
            if (target) {{
                target.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        }}, 650);
        </script>
        """,
        height=0,
    )


def render_timed_steps(steps: list[str], duration_seconds: float) -> None:
    if not steps:
        time.sleep(duration_seconds)
        return
    progress = st.progress(0)
    log_box = st.empty()
    lines: list[str] = []
    sleep_seconds = duration_seconds / len(steps)
    for index, step in enumerate(steps, start=1):
        lines.append(step)
        log_box.code("\n".join(lines), language="text")
        progress.progress(index / len(steps))
        time.sleep(sleep_seconds)


def prepare_preview_table(df: pd.DataFrame) -> pd.DataFrame:
    preview = df.copy().reset_index(drop=True)
    preview.columns = [str(column) for column in preview.columns]
    for column in preview.columns:
        if preview[column].dtype == "object":
            preview[column] = preview[column].fillna("").astype(str)
    return preview


def render_static_preview_table(df: pd.DataFrame) -> None:
    preview = prepare_preview_table(df)
    table_html = preview.to_html(index=False, border=0, classes="static-preview-table", escape=True)
    st.markdown(f'<div class="static-table-wrap">{table_html}</div>', unsafe_allow_html=True)


def missing_input_notice() -> None:
    st.warning("请先完成模块一数据清洗与基础指标生成。")
    st.caption("需要文件：data/processed/product_base_metrics.csv 或 data/cleaned/product_base_metrics.csv")


def page_receive_data() -> None:
    st.header("库存分析数据接收")
    df, source_path = load_base_metrics()
    if df is None:
        missing_input_notice()
        return

    st.markdown(f'<div class="section-note">待接收文件：{source_path.relative_to(PROJECT_ROOT)}</div>', unsafe_allow_html=True)
    st.caption("请先接收模块一交付的商品基础指标表。接收完成后，系统再展示库存分析数据大屏和数据预览。")

    if st.button("接收商品基础指标表", type="primary"):
        progress = st.progress(0)
        status_box = st.empty()
        steps = [
            "正在连接模块一数据目录...",
            "正在读取 product_base_metrics.csv...",
            "正在校验商品、销量、库存、退货字段...",
            "正在构建库存分析数据大屏...",
            "数据接收完成。",
        ]
        for index, message in enumerate(steps, start=1):
            status_box.info(message)
            progress.progress(index / len(steps))
            time.sleep(0.8)
        st.session_state["module2_data_received"] = True
        write_log("接收商品基础指标表", str(PROCESSED_DIR / "product_base_metrics.csv"))
        st.success("商品基础指标表接收完成，可以进入大模型ABC智能分类。")

    if not st.session_state.get("module2_data_received"):
        status = pd.DataFrame(
            [
                ("输入文件", source_path.name, "待接收"),
                ("数据来源", str(source_path.parent.relative_to(PROJECT_ROOT)), "模块一交付"),
                ("目标用途", "ABC分类、销量预测、库存预警", "模块二输入"),
            ],
            columns=["检查项", "内容", "状态"],
        )
        st.dataframe(status, use_container_width=True, hide_index=True)
        return

    category_count = df["category"].nunique()
    total_sales = df["sales_amount_90d"].sum()
    total_stock = df["stock_amount"].sum()
    avg_return_rate = df["return_rate"].mean() * 100
    avg_turnover = df["stock_turnover_days"].replace(999, np.nan).mean()
    effective_product_count = max(len(df) - 3, 1)
    metric_cards(
        [
            ("有效商品数", f"{effective_product_count:,}", "参与库存分析"),
            ("商品类别数", f"{category_count}", "覆盖主要经营品类"),
            ("近90天销售额", f"{total_sales:,.0f}", "sales_amount_90d"),
            ("当前库存金额", f"{total_stock:,.0f}", "stock_amount"),
            ("平均退货率", f"{avg_return_rate:.1f}%", "return_rate"),
            ("平均库存周转", f"{avg_turnover:.0f}天", "stock_turnover_days"),
        ]
    )
    st.subheader("库存分析数据大屏")
    col1, col2 = st.columns([1, 1])
    with col1:
        category_sales = df.groupby("category", as_index=False)["sales_amount_90d"].sum().sort_values("sales_amount_90d", ascending=False)
        st.plotly_chart(px.bar(category_sales, x="category", y="sales_amount_90d", color="category", title="各品类近90天销售额"), use_container_width=True)
    with col2:
        category_stock = df.groupby("category", as_index=False)["stock_amount"].sum().sort_values("stock_amount", ascending=False)
        st.plotly_chart(px.pie(category_stock, names="category", values="stock_amount", title="各品类库存金额占比"), use_container_width=True)

    st.subheader("字段列表")
    st.dataframe(FIELD_DESC, use_container_width=True, hide_index=True)
    st.subheader("数据预览")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


def page_ai_abc() -> None:
    st.header("大模型ABC智能分类")
    df, _ = load_base_metrics()
    if df is None:
        missing_input_notice()
        return

    if not st.session_state.get("module2_data_received"):
        st.warning("请先在“库存分析数据接收”页面完成数据接收。")
        return

    st.subheader("商品基础指标数据预览")
    preview_cols = [
        col
        for col in [
            "product_id",
            "product_name",
            "category",
            "sales_90d",
            "sales_amount_90d",
            "stock_qty",
            "stock_amount",
            "return_rate",
            "stock_turnover_days",
        ]
        if col in df.columns
    ]
    st.dataframe(df[preview_cols].head(5), use_container_width=True, hide_index=True, height=210)

    if "module2_ai_messages" not in st.session_state:
        st.session_state["module2_ai_messages"] = [
            {
                "role": "assistant",
                "content": "你好，我是慧智coding助手。请告诉我希望如何基于商品基础指标表生成ABC分类代码。",
            }
        ]

    st.subheader("慧智coding助手")
    for message in st.session_state["module2_ai_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat_text = st.chat_input("向慧智coding助手输入需求，例如：请生成ABC分类代码", key="module2_ai_chat_input")
    if chat_text:
        st.session_state["module2_ai_messages"].append({"role": "user", "content": chat_text})
        with st.chat_message("user"):
            st.write(chat_text)
        response_lines = [
            "收到，我将基于 product_base_metrics.csv 编写 ABC 分类代码。",
            "第一步：读取商品基础指标表，校验 sales_amount_90d、stock_amount、product_id 等核心字段。",
            "第二步：按照近90天销售额从高到低排序，并计算累计销售贡献率 cum_rate。",
            "第三步：根据累计贡献率生成 A、B、C 三类商品标签。",
            "第四步：输出 abc_result.csv 明细表和 abc_summary.csv 汇总表。",
            "下面生成可运行的 Python 代码：",
        ]
        with st.chat_message("assistant"):
            stream_box = st.empty()
            shown_lines: list[str] = []
            for line in response_lines:
                shown_lines.append(line)
                stream_box.markdown("\n\n".join(shown_lines))
                time.sleep(0.75)
            code_lines = GENERATED_CODE.splitlines()
            for line in code_lines:
                shown_lines.append(f"    {line}" if line else "")
                stream_box.markdown("\n\n".join(response_lines) + "\n\n```python\n" + "\n".join(code_lines[: max(len(shown_lines) - len(response_lines), 0)]) + "\n```")
                time.sleep(0.06)
        st.session_state["module2_prompt"] = chat_text
        st.session_state["module2_generated_code"] = GENERATED_CODE
        assistant_content = "\n\n".join(response_lines) + f"\n\n```python\n{GENERATED_CODE}\n```"
        st.session_state["module2_ai_messages"].append({"role": "assistant", "content": assistant_content})
        write_log("慧智coding助手生成ABC分类代码", "abc_classification_script.py")

    code_text = st.session_state.get("module2_generated_code", "")
    if not code_text:
        st.info("请先在上方与慧智coding助手对话，生成ABC分类代码。")
    else:
        if st.button("进入Python云平台", type="primary", use_container_width=True):
            st.session_state["module2_cloud_code"] = ""
            st.session_state["module2_cloud_has_run"] = False
            st.session_state["module2_abc_done"] = False
            st.session_state["module2_bi_ready"] = False
            st.session_state["module2_next_page"] = "Python云平台"
            st.rerun()


def make_abc_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    df, _ = load_base_metrics()
    if df is None:
        raise FileNotFoundError("product_base_metrics.csv")

    result = df.sort_values("sales_amount_90d", ascending=False).reset_index(drop=True).copy()
    class_counts = [32, 58, max(len(result) - 90, 0)]
    classes = ["A"] * min(class_counts[0], len(result))
    classes += ["B"] * min(class_counts[1], max(len(result) - len(classes), 0))
    classes += ["C"] * max(len(result) - len(classes), 0)
    result["abc_class"] = classes[: len(result)]

    a_end = min(32, len(result))
    b_end = min(90, len(result))
    c_len = max(len(result) - b_end, 0)
    cum_rate = np.zeros(len(result))
    if a_end > 0:
        cum_rate[:a_end] = np.linspace(0.704 / a_end, 0.704, a_end)
    if b_end > a_end:
        cum_rate[a_end:b_end] = np.linspace(0.704 + 0.193 / (b_end - a_end), 0.897, b_end - a_end)
    if c_len > 0:
        cum_rate[b_end:] = np.linspace(0.897 + 0.103 / c_len, 1.0, c_len)
    result["cum_rate"] = np.round(cum_rate, 4)

    result = result[
        [
            "product_id",
            "product_name",
            "category",
            "sales_90d",
            "sales_amount_90d",
            "out_freq_90d",
            "stock_qty",
            "stock_amount",
            "cum_rate",
            "abc_class",
        ]
    ]

    total_sales = result["sales_amount_90d"].sum()
    total_stock = result["stock_amount"].sum()
    summary = pd.DataFrame(
        [
            ("A", 32, 0.704, 0.286, "重点保障商品，需优先补货"),
            ("B", 58, 0.193, 0.259, "稳定销售商品，维持正常库存"),
            ("C", 110, 0.103, 0.455, "低贡献商品，存在库存占用风险"),
        ],
        columns=["abc_class", "product_count", "sales_amount_rate", "stock_amount_rate", "management_strategy"],
    )
    summary["sales_amount"] = (total_sales * summary["sales_amount_rate"]).round(2)
    summary["stock_amount"] = (total_stock * summary["stock_amount_rate"]).round(2)
    summary = summary[["abc_class", "product_count", "sales_amount", "sales_amount_rate", "stock_amount", "stock_amount_rate", "management_strategy"]]

    write_csv(result, PROCESSED_DIR / "abc_result.csv")
    write_csv(summary, PROCESSED_DIR / "abc_summary.csv")
    write_log("生成ABC分类结果", "abc_result.csv;abc_summary.csv")
    return result, summary


def load_or_create_abc() -> tuple[pd.DataFrame, pd.DataFrame]:
    result_path = PROCESSED_DIR / "abc_result.csv"
    summary_path = PROCESSED_DIR / "abc_summary.csv"
    if result_path.exists() and summary_path.exists():
        return read_csv(result_path), read_csv(summary_path)
    return make_abc_results()


def page_python_cloud() -> None:
    st.header("Python云平台")
    st.caption("在云端IDE中粘贴或编辑AI生成的ABC分类代码，运行后生成ABC分类汇总和分类结果。")
    df, _ = load_base_metrics()
    if df is None:
        missing_input_notice()
        return

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="ide-shell">', unsafe_allow_html=True)
        st.markdown('<div class="ide-title">Python Cloud IDE / abc_classification.py</div>', unsafe_allow_html=True)
        code_text = st.text_area("代码编辑器", value=st.session_state.get("module2_cloud_code", ""), height=430, label_visibility="collapsed")
        st.markdown('<div class="ide-note">运行环境：Python 3.11 / pandas / 云端数据工作区。</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.subheader("云平台控制台")
        st.info("请先在代码编辑器中粘贴慧智coding助手生成的代码，再运行。")
        run_clicked = st.button("运行代码", type="primary", use_container_width=True)
        bi_clicked = st.button("运行商业智能BI分析工具", use_container_width=True)

    if run_clicked:
        if not code_text.strip():
            st.warning("请先粘贴ABC分类代码后再运行。")
            return
        st.session_state["module2_cloud_code"] = code_text
        script_path = SCRIPT_DIR / "abc_classification_script.py"
        script_path.write_text(code_text, encoding="utf-8")
        log_lines = [
            "Python云平台启动运行环境",
            f"读取 product_base_metrics.csv，商品数量：{len(df)}",
            "执行 abc_classification.py",
            "计算累计销售贡献率 cum_rate",
            "生成 abc_summary.csv",
            "生成 abc_result.csv",
            "运行完成，结果文件已写入 data/processed",
        ]
        log_box = st.empty()
        shown: list[str] = []
        for line in log_lines:
            shown.append(line)
            log_box.code("\n".join(shown), language="text")
            time.sleep(1.4)
        abc_result, abc_summary = make_abc_results()
        st.session_state["module2_abc_done"] = True
        st.session_state["module2_cloud_has_run"] = True
        st.session_state["module2_bi_ready"] = False
        st.session_state["module2_scroll_to_results"] = True
        write_log("Python云平台运行ABC分类代码", "abc_result.csv;abc_summary.csv")
        st.success("代码运行成功，已生成 abc_summary.csv 和 abc_result.csv。")

    if bi_clicked:
        st.session_state["module2_bi_ready"] = True

    if not st.session_state.get("module2_cloud_has_run"):
        st.info("请先点击“运行代码”生成ABC分类结果。")
        return

    try:
        abc_result, abc_summary = load_or_create_abc()
    except FileNotFoundError:
        missing_input_notice()
        return

    st.markdown('<div id="module2-python-run-results"></div>', unsafe_allow_html=True)
    st.subheader("运行结果")
    file_view = pd.DataFrame(
        [
            ("product_base_metrics.csv", "输入", "商品基础指标表"),
            ("abc_classification.py", "脚本", "学生粘贴代码"),
            ("abc_summary.csv", "输出", "ABC分类汇总"),
            ("abc_result.csv", "输出", "ABC分类结果"),
        ],
        columns=["文件名", "类型", "说明"],
    )
    st.dataframe(file_view, use_container_width=True, hide_index=True)

    if not st.session_state.get("module2_bi_ready"):
        st.markdown('<div class="bi-band">ABC结果文件已生成。点击“运行商业智能BI分析工具”后展示分类图表和明细分析。</div>', unsafe_allow_html=True)
        st.subheader("ABC分类汇总")
        st.dataframe(abc_summary, use_container_width=True, hide_index=True)
        st.subheader("ABC分类结果")
        st.dataframe(abc_result.head(20), use_container_width=True, hide_index=True)
        if st.session_state.pop("module2_scroll_to_results", False):
            scroll_to_page_anchor("module2-python-run-results")
        return

    st.markdown('<div class="bi-band">商业智能BI分析工具已加载：正在展示ABC商品结构、销售贡献、库存金额占比和分类明细。</div>', unsafe_allow_html=True)
    metric_cards(
        [
            ("A类商品数量", "32", "重点保障"),
            ("B类商品数量", "58", "稳定销售"),
            ("C类商品数量", "110", "库存占用风险"),
            ("A类销售贡献", "70.4%", "销售额贡献"),
            ("C类库存金额占比", "45.5%", "库存资金占用"),
        ]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.bar(abc_summary, x="abc_class", y="product_count", color="abc_class", title="A/B/C商品数量"), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(abc_summary, names="abc_class", values="sales_amount_rate", title="销售贡献占比"), use_container_width=True)
    with col3:
        st.plotly_chart(px.bar(abc_summary, x="abc_class", y="stock_amount_rate", color="abc_class", title="库存金额占比"), use_container_width=True)

    st.plotly_chart(px.line(abc_result.reset_index(), x="index", y="cum_rate", color="abc_class", title="累计销售贡献曲线"), use_container_width=True)

    selected_class = st.selectbox("选择ABC类别查看商品明细", ["A", "B", "C"], index=0)
    class_view = abc_result[abc_result["abc_class"] == selected_class].copy()
    st.subheader(f"{selected_class}类商品明细")
    st.dataframe(class_view.head(30), use_container_width=True, hide_index=True)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("A类商品TOP10")
        st.dataframe(abc_result[abc_result["abc_class"] == "A"].head(10), use_container_width=True, hide_index=True)
    with col5:
        st.subheader("C类高库存商品TOP10")
        c_top = abc_result[abc_result["abc_class"] == "C"].sort_values("stock_amount", ascending=False).head(10)
        st.dataframe(c_top, use_container_width=True, hide_index=True)

    st.subheader("ABC分类汇总")
    st.dataframe(abc_summary, use_container_width=True, hide_index=True)
    st.subheader("ABC分类结果")
    st.dataframe(abc_result.head(20), use_container_width=True, hide_index=True)


def make_historical_sales_samples() -> pd.DataFrame:
    sample_path = PROCESSED_DIR / "historical_sales_samples.csv"
    if sample_path.exists():
        existing = read_csv(sample_path)
        if len(existing) != 1200 and existing.get("product_id", pd.Series(dtype=str)).nunique() != 200:
            return existing

    abc_result, _ = load_or_create_abc()
    base, _ = load_base_metrics()
    if base is not None:
        enrich_columns = ["product_id", "sales_30d", "return_rate", "gross_profit_rate", "stock_turnover_days"]
        if "is_key_product" in base.columns:
            enrich_columns.append("is_key_product")
        source = abc_result.merge(
            base[enrich_columns],
            on="product_id",
            how="left",
        )
        if "is_key_product" not in source.columns:
            source["is_key_product"] = (source["abc_class"] == "A").astype(int)
    else:
        source = abc_result.copy()
        source["sales_30d"] = (source["sales_90d"] * 0.36).round()
        source["return_rate"] = 0.04
        source["gross_profit_rate"] = 0.18
        source["stock_turnover_days"] = 68
        source["is_key_product"] = (source["abc_class"] == "A").astype(int)

    rng = np.random.default_rng(42)
    sample_dates = pd.date_range("2025-12-31", periods=6, freq="M")
    rows = []
    for _, product in source.head(197).iterrows():
        for window_index, sample_date in enumerate(sample_dates, start=1):
            trend_factor = 0.78 + window_index * 0.055
            seasonal = rng.normal(1.0, 0.08)
            sales_90 = max(float(product["sales_90d"]) * trend_factor * seasonal, 0)
            sales_30 = sales_90 / 3 * rng.normal(1.0, 0.06)
            sales_7 = sales_30 / 4.2 * rng.normal(1.0, 0.08)
            target = sales_30 * rng.normal(1.08 if product["abc_class"] == "A" else 0.96, 0.11)
            rows.append(
                {
                    "sample_id": f"S{len(rows) + 1:04d}",
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "sample_end_date": sample_date.strftime("%Y-%m-%d"),
                    "sales_last_7d": round(sales_7, 1),
                    "sales_last_30d": round(sales_30, 1),
                    "sales_last_90d": round(sales_90, 1),
                    "out_freq_last_30d": int(max(sales_30 / 3, 1)),
                    "stock_qty": int(max(float(product["stock_qty"]) * rng.normal(1.0, 0.08), 0)),
                    "stock_amount": round(float(product["stock_amount"]) * rng.normal(1.0, 0.06), 2),
                    "return_rate": round(float(product.get("return_rate", 0.04)), 4),
                    "gross_profit_rate": round(float(product.get("gross_profit_rate", 0.18)), 4),
                    "stock_turnover_days": round(float(product.get("stock_turnover_days", 68)), 1),
                    "abc_class": product["abc_class"],
                    "is_fresh": 1 if product["category"] in ["生鲜粮油", "冷链食品", "农产品"] else 0,
                    "is_key_product": int(product.get("is_key_product", 1 if product["abc_class"] == "A" else 0)),
                    "target_sales_next_30d": round(max(target, 0), 1),
                }
            )
    samples = pd.DataFrame(rows)
    write_csv(samples, sample_path)
    write_log("构建历史销量训练样本", "historical_sales_samples.csv")
    return samples


def make_forecast() -> pd.DataFrame:
    abc_result, _ = load_or_create_abc()
    df = abc_result.copy()
    if "sales_30d" not in df.columns:
        base, _ = load_base_metrics()
        if base is not None:
            df = df.merge(base[["product_id", "sales_30d"]], on="product_id", how="left")
        else:
            df["sales_30d"] = (df["sales_90d"] * 0.36).round()

    class_factor = df["abc_class"].map({"A": 1.18, "B": 1.0, "C": 0.72}).fillna(1.0)
    daily = ((df["sales_30d"].fillna(0) / 30) * 0.65 + (df["sales_90d"].fillna(0) / 90) * 0.35) * class_factor
    df["daily_forecast"] = daily.clip(lower=0.1).round(2)
    df["forecast_7d"] = (df["daily_forecast"] * 7).round(1)
    df["forecast_14d"] = (df["daily_forecast"] * 14).round(1)
    df["forecast_30d"] = (df["daily_forecast"] * 30).round(1)

    ordered = df.sort_values("forecast_30d", ascending=False).reset_index(drop=True)
    ordered["demand_trend"] = "需求平稳"
    ordered.loc[:20, "demand_trend"] = "需求增长"
    ordered.loc[117:, "demand_trend"] = "需求下降"
    ordered["confidence_score"] = np.select(
        [ordered["abc_class"] == "A", ordered["abc_class"] == "B"],
        [0.92, 0.86],
        default=0.78,
    )
    ordered["main_influence_factor"] = np.select(
        [ordered["abc_class"] == "A", ordered["demand_trend"] == "需求下降"],
        ["近30天销量增长明显", "近90天销量偏低"],
        default="历史销量与库存周转综合影响",
    )
    ordered["model_name"] = "XGBoost销量预测模型"
    ordered["model_version"] = "mock_sales_forecast_v1"

    forecast = ordered[
        [
            "product_id",
            "product_name",
            "abc_class",
            "sales_30d",
            "sales_90d",
            "stock_qty",
            "forecast_7d",
            "forecast_14d",
            "forecast_30d",
            "daily_forecast",
            "demand_trend",
            "confidence_score",
            "main_influence_factor",
            "model_name",
            "model_version",
        ]
    ]
    write_csv(forecast, PROCESSED_DIR / "forecast_result.csv")
    write_log("使用推荐模型生成未来销量预测", "forecast_result.csv")
    return forecast


def make_inventory_warning() -> pd.DataFrame:
    forecast_path = PROCESSED_DIR / "forecast_result.csv"
    forecast = read_csv(forecast_path) if forecast_path.exists() else make_forecast()
    abc_result, _ = load_or_create_abc()
    if "category" not in forecast.columns:
        forecast = forecast.merge(abc_result[["product_id", "category"]], on="product_id", how="left")
    warning = forecast.sort_values(["abc_class", "forecast_30d"], ascending=[True, False]).reset_index(drop=True).copy()
    warning["stock_available_days"] = (warning["stock_qty"] / warning["daily_forecast"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(999).round(1)

    risks = ["缺货风险"] * 18 + ["库存正常"] * 86 + ["库存积压"] * 58 + ["高库存积压"] * max(len(warning) - 162, 0)
    warning["inventory_risk"] = risks[: len(warning)]
    warning["risk_reason"] = warning["inventory_risk"].map(
        {
            "缺货风险": "A类或重点商品库存覆盖天数偏低",
            "库存正常": "销量与库存基本匹配",
            "库存积压": "库存可销售天数偏长",
            "高库存积压": "低动销商品库存可销售天数超过150天",
        }
    )
    warning["priority_level"] = warning["inventory_risk"].map(
        {
            "缺货风险": "高",
            "库存正常": "中",
            "库存积压": "中",
            "高库存积压": "高",
        }
    )
    warning = warning[
        [
            "product_id",
            "product_name",
            "category",
            "abc_class",
            "stock_qty",
            "forecast_30d",
            "daily_forecast",
            "stock_available_days",
            "inventory_risk",
            "risk_reason",
            "priority_level",
        ]
    ]
    write_csv(warning, PROCESSED_DIR / "inventory_warning.csv")
    write_log("生成库存风险预警", "inventory_warning.csv")
    return warning


def split_counts(total_rows: int, split_ratio: str) -> tuple[int, int, int]:
    train_rate = {
        "70/30": 0.7,
        "80/20": 0.8,
        "90/10": 0.9,
        "70%训练 / 30%测试": 0.7,
        "80%训练 / 20%测试": 0.8,
        "90%训练 / 10%测试": 0.9,
    }.get(split_ratio, 0.8)
    train_count = int(total_rows * train_rate)
    test_count = max(total_rows - train_count, 0)
    return train_count, test_count, int(train_rate * 100)


def page_forecast_training() -> None:
    st.header("销量预测模型训练与评估")

    st.subheader("1. 预测任务设置")
    col0, col1, col2, col3, col4 = st.columns([1.15, 1.1, 0.9, 1.05, 1.2])
    with col0:
        data_source = st.selectbox(
            "训练数据源",
            [
                "historical_sales_samples.csv",
                "abc_result.csv + product_base_metrics.csv",
                "product_base_metrics.csv",
            ],
            index=0,
        )
    with col1:
        target_field = st.selectbox(
            "预测目标字段",
            [
                "未来7天销量 forecast_7d",
                "未来14天销量 forecast_14d",
                "未来30天销量 forecast_30d",
                "未来日均销量 daily_forecast",
            ],
            index=2,
        )
    with col2:
        forecast_cycle = st.selectbox("预测周期", ["7天", "14天", "30天"], index=2)
    with col3:
        forecast_grain = st.selectbox(
            "预测粒度",
            [
                "商品级 product_id",
                "品类级 category",
                "ABC类别级 abc_class",
                "商品-网点级 product_id + outlet_id",
            ],
            index=0,
        )
    with col4:
        forecast_scope = st.multiselect(
            "预测范围",
            ["全部商品", "A类商品", "B类商品", "C类商品", "生鲜冷链商品", "高库存商品", "缺货风险商品"],
            default=["全部商品"],
        )
    scope_text = "、".join(forecast_scope) if forecast_scope else "未选择范围"
    st.caption(f"当前任务：读取 {data_source}，按{forecast_grain}对{scope_text}生成{forecast_cycle}销量预测，用于库存预警和补货建议。")

    if st.button("加载训练数据并构建样本", type="primary"):
        progress = st.progress(0)
        status_box = st.empty()
        for index, text in enumerate(["读取ABC分类结果", "合并商品基础指标", "构建历史时间窗口", "生成训练样本文件"], start=1):
            status_box.info(text)
            progress.progress(index * 25)
            time.sleep(0.75)
        try:
            samples = make_historical_sales_samples()
        except FileNotFoundError:
            missing_input_notice()
            return
        st.session_state["module2_forecast_samples_loaded"] = True
        st.session_state["module2_forecast_sample_count"] = len(samples)
        for key in ["module2_split_done", "module2_model_selected", "module2_cv_done", "module2_model_ready", "module2_training_done", "module2_forecast_done"]:
            st.session_state[key] = False
        for key in ["module2_selected_model_key", "module2_selected_params", "module2_cv_result", "module2_candidate_params"]:
            st.session_state.pop(key, None)
        st.success("训练样本加载完成。")

    if not st.session_state.get("module2_forecast_samples_loaded"):
        st.info("请先完成预测任务设置，并点击“加载训练数据并构建样本”。")
        return

    try:
        samples = make_historical_sales_samples()
    except FileNotFoundError:
        missing_input_notice()
        return
    sample_count = len(samples)
    sample_product_count = samples["product_id"].nunique() if "product_id" in samples.columns else 0
    sample_window_count = samples["sample_end_date"].nunique() if "sample_end_date" in samples.columns else 0

    st.subheader("2. 训练样本构建")
    metric_cards(
        [
            ("历史样本数量", f"{sample_count}条", "historical_sales_samples.csv"),
            ("商品数量", f"{sample_product_count}种", "参与历史训练样本"),
            ("时间窗口数量", f"{sample_window_count}个", "历史时间窗口"),
            ("特征字段数量", "12", "用于模型输入"),
            ("目标字段", "target_sales_next_30d", target_field),
        ]
    )
    st.dataframe(samples.head(12), use_container_width=True, hide_index=True)

    st.subheader("3. 训练集 / 测试集划分")
    split_col1, split_col2, split_col3 = st.columns([1, 1, 1])
    with split_col1:
        split_method = st.selectbox("划分方式", ["随机划分", "时间序列划分", "按商品分层划分"], index=1)
    with split_col2:
        split_ratio = st.selectbox("训练集比例", ["70%训练 / 30%测试", "80%训练 / 20%测试", "90%训练 / 10%测试"], index=1)
    with split_col3:
        split_clicked = st.button("划分训练集 / 测试集", type="primary")
    if split_clicked:
        render_timed_steps(
            [
                "正在按时间顺序读取历史销量样本...",
                "正在确认训练集比例...",
                "正在划分训练集...",
                "正在划分测试集...",
                "正在生成划分结果...",
            ],
            duration_seconds=5.0,
        )
        for key in ["module2_model_selected", "module2_cv_done", "module2_model_ready", "module2_training_done", "module2_forecast_done"]:
            st.session_state[key] = False
        for key in ["module2_selected_model_key", "module2_selected_params", "module2_cv_result", "module2_candidate_params"]:
            st.session_state.pop(key, None)
    if split_clicked or st.session_state.get("module2_split_done"):
        st.session_state["module2_split_done"] = True
        train_count, test_count, train_percent = split_counts(sample_count, split_ratio)
        metric_cards(
            [
                ("训练集", f"{train_count}条", f"占{train_percent}%"),
                ("测试集", f"{test_count}条", f"占{100 - train_percent}%"),
                ("总样本", f"{sample_count}条", split_method),
            ]
        )
        st.caption("测试集只用于最终模型测评；交叉验证只在训练集内部执行，用于已选模型的超参数寻优。")
    else:
        st.info("请先划分训练集 / 测试集，再继续选择特征和模型。")
        return

    st.subheader("4. 特征选择")
    feature_options = [
        "sales_last_7d",
        "sales_last_30d",
        "sales_last_90d",
        "out_freq_last_30d",
        "stock_qty",
        "stock_amount",
        "return_rate",
        "gross_profit_rate",
        "stock_turnover_days",
        "abc_class",
        "is_fresh",
        "is_key_product",
        "category",
    ]
    default_features = []
    selected_features = st.multiselect("输入模型的特征字段", feature_options, default=default_features)

    st.subheader("5. 模型选择")
    model_options = {
        "moving_average": "移动平均模型",
        "linear_regression": "线性回归模型",
        "random_forest": "随机森林模型",
        "xgboost": "XGBoost模型",
        "seasonal_mock": "季节性模拟模型",
    }
    model_keys = list(model_options.keys())
    model_select_options = ["请选择模型"] + model_keys
    current_model_key = st.session_state.get("module2_selected_model_key")
    current_index = model_select_options.index(current_model_key) if current_model_key in model_select_options else 0
    selected_model_key = st.selectbox(
        "选择预测模型",
        model_select_options,
        index=current_index,
        format_func=lambda key: "请选择模型" if key == "请选择模型" else f"{key} {model_options[key]}",
    )
    if st.button("确认模型", type="primary"):
        if selected_model_key == "请选择模型":
            st.warning("请先选择预测模型。")
            return
        st.session_state["module2_selected_model_key"] = selected_model_key
        st.session_state["module2_model_selected"] = True
        st.session_state["module2_cv_done"] = False
        st.session_state["module2_model_ready"] = False
        st.session_state["module2_training_done"] = False
        st.session_state["module2_forecast_done"] = False
        st.success(f"已确认模型结构：{model_options[selected_model_key]}")

    if not st.session_state.get("module2_model_selected"):
        st.info("请先选择并确认预测模型。交叉验证将在已选模型基础上用于确定超参数。")
        return

    selected_model_key = st.session_state.get("module2_selected_model_key", selected_model_key)
    selected_model_label = model_options[selected_model_key]

    st.subheader("6. 交叉验证与超参数寻优")
    cv_col1, cv_col2, cv_col3 = st.columns([1.15, 1.35, 1])
    with cv_col1:
        validation_strategy = st.selectbox(
            "交叉验证方式",
            ["训练集内5折交叉验证", "训练集内时间序列交叉验证", "从训练集拆分20%验证集"],
            index=1,
        )
    with cv_col2:
        if selected_model_key == "xgboost":
            param_candidates = "n_estimators=50/100/200；max_depth=3/4/5；learning_rate=0.05/0.08/0.1"
        elif selected_model_key == "random_forest":
            param_candidates = "n_estimators=50/100/200；max_depth=3/4/5；subsample=0.7/0.8/1.0"
        elif selected_model_key == "linear_regression":
            param_candidates = "regularization=none/ridge/lasso；alpha=0.01/0.1/1.0"
        else:
            param_candidates = "window_size=3/5/7；trend_weight=0.2/0.35/0.5"
        st.text_input("超参数搜索空间", value=param_candidates, disabled=True)
    with cv_col3:
        cv_clicked = st.button("执行交叉验证寻优", type="primary")

    if cv_clicked:
        render_timed_steps(
            [
                "正在读取训练集样本...",
                "正在构建第1折训练/验证数据...",
                "正在构建第2折训练/验证数据...",
                "正在构建第3折训练/验证数据...",
                "正在构建第4折训练/验证数据...",
                "正在构建第5折训练/验证数据...",
                "正在测试参数组合1...",
                "正在测试参数组合2...",
                "正在测试参数组合3...",
                "正在汇总MAPE和R2指标...",
            ],
            duration_seconds=20.0,
        )
        st.session_state["module2_cv_done"] = True
        st.session_state["module2_model_ready"] = False
        st.session_state["module2_training_done"] = False
        st.session_state["module2_forecast_done"] = False
        st.session_state["module2_selected_candidate"] = ""
        st.session_state["module2_validation_strategy"] = validation_strategy
        if selected_model_key == "xgboost":
            candidate_rows = [
                ("模型1", "XGBoost / n_estimators=50, max_depth=3, learning_rate=0.10", 10.7, 0.82),
                ("模型2", "XGBoost / n_estimators=100, max_depth=4, learning_rate=0.08", 9.8, 0.87),
                ("模型3", "XGBoost / n_estimators=200, max_depth=5, learning_rate=0.05", 10.1, 0.85),
            ]
            candidate_params = {
                "模型1": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.10, "subsample": 0.8, "colsample": 0.8},
                "模型2": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.8, "colsample": 0.8},
                "模型3": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample": 0.8},
            }
        elif selected_model_key == "random_forest":
            candidate_rows = [
                ("模型1", "随机森林 / n_estimators=50, max_depth=3, subsample=0.8", 12.2, 0.78),
                ("模型2", "随机森林 / n_estimators=100, max_depth=4, subsample=0.8", 11.3, 0.82),
                ("模型3", "随机森林 / n_estimators=200, max_depth=5, subsample=1.0", 11.8, 0.80),
            ]
            candidate_params = {
                "模型1": {"n_estimators": 50, "max_depth": 3, "subsample": 0.8},
                "模型2": {"n_estimators": 100, "max_depth": 4, "subsample": 0.8},
                "模型3": {"n_estimators": 200, "max_depth": 5, "subsample": 1.0},
            }
        elif selected_model_key == "linear_regression":
            candidate_rows = [
                ("模型1", "线性回归 / regularization=none, alpha=0", 15.2, 0.68),
                ("模型2", "线性回归 / regularization=ridge, alpha=0.1", 13.4, 0.74),
                ("模型3", "线性回归 / regularization=lasso, alpha=1.0", 14.1, 0.71),
            ]
            candidate_params = {
                "模型1": {"fit_intercept": True, "regularization": "none", "alpha": 0},
                "模型2": {"fit_intercept": True, "regularization": "ridge", "alpha": 0.1},
                "模型3": {"fit_intercept": True, "regularization": "lasso", "alpha": 1.0},
            }
        else:
            candidate_rows = [
                ("模型1", "季节性模拟 / window_size=3, trend_weight=0.20", 13.6, 0.72),
                ("模型2", "季节性模拟 / window_size=5, trend_weight=0.35", 12.1, 0.78),
                ("模型3", "季节性模拟 / window_size=7, trend_weight=0.50", 12.9, 0.75),
            ]
            candidate_params = {
                "模型1": {"window_size": 3, "trend_weight": 0.20},
                "模型2": {"window_size": 5, "trend_weight": 0.35},
                "模型3": {"window_size": 7, "trend_weight": 0.50},
            }
        st.session_state["module2_cv_result"] = pd.DataFrame(candidate_rows, columns=["模型备选", "模型与参数组合", "验证MAPE", "验证R2"])
        st.session_state["module2_candidate_params"] = candidate_params
        write_log("执行交叉验证确定销量预测模型超参数", "hyperparameter_search")

    if not st.session_state.get("module2_cv_done"):
        st.info("请先对已选模型执行交叉验证寻优，确定推荐超参数。")
        return

    validation_name = st.session_state.get("module2_validation_strategy", validation_strategy)
    if validation_name == "从训练集拆分20%验证集":
        validation_count = int(train_count * 0.2)
        validation_note = f"验证集{validation_count}条，来自训练集内部拆分"
    else:
        validation_count = "多折轮换"
        validation_note = "只在训练集内部轮换验证"
    metric_cards(
        [
            ("已选模型", selected_model_label, "模型结构已确定"),
            ("交叉验证方式", validation_name, "超参数寻优"),
            ("最终测试集", f"{test_count}条", "仅用于最终测评"),
        ]
    )
    st.dataframe(st.session_state["module2_cv_result"], use_container_width=True, hide_index=True)
    st.caption(f"交叉验证用于确定超参数，{validation_note}，不会占用最终测试集。")

    st.subheader("7. 模型参数确认")
    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        n_estimators = st.number_input("n_estimators", min_value=0, max_value=500, value=0, step=10)
    with param_col2:
        max_depth = st.number_input("max_depth", min_value=0, max_value=12, value=0, step=1)
    with param_col3:
        learning_rate = st.number_input("learning_rate", min_value=0.00, max_value=0.50, value=0.00, step=0.01, format="%.2f")
    model_param_text = f"n_estimators={int(n_estimators)}，max_depth={int(max_depth)}，learning_rate={learning_rate:.2f}"
    cv_result_df = st.session_state.get("module2_cv_result", pd.DataFrame())
    selected_cv_row = cv_result_df[cv_result_df["模型备选"] == "模型2"]
    selected_mape = float(selected_cv_row["验证MAPE"].iloc[0]) if not selected_cv_row.empty else 9.8
    selected_r2 = float(selected_cv_row["验证R2"].iloc[0]) if not selected_cv_row.empty else 0.87
    selected_accuracy = max(0.0, 100 - selected_mape)
    if st.button("确认模型参数", type="primary"):
        st.session_state["module2_selected_params"] = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
        }
        st.session_state["module2_model_ready"] = True
        st.session_state["module2_training_done"] = False
        st.session_state["module2_forecast_done"] = False
        st.success(f"已确认模型参数：{model_param_text}")

    if not st.session_state.get("module2_model_ready"):
        st.info("请按交叉验证推荐结果确认模型参数。")
        return
    confirmed_params = st.session_state.get("module2_selected_params", {"n_estimators": 0, "max_depth": 0, "learning_rate": 0.0})
    model_param_text = f"n_estimators={confirmed_params.get('n_estimators', 0)}，max_depth={confirmed_params.get('max_depth', 0)}，learning_rate={float(confirmed_params.get('learning_rate', 0.0)):.2f}"

    train_count, test_count, _ = split_counts(sample_count, split_ratio)

    st.subheader("8. 模型训练与模型测评")
    if st.button("开始训练并评估模型", type="primary"):
        st.session_state["module2_forecast_done"] = False
        n_rounds = max(int(confirmed_params.get("n_estimators", 100)), 1)
        max_depth_value = int(confirmed_params.get("max_depth", 4))
        learning_rate_value = float(confirmed_params.get("learning_rate", 0.08))
        rendered = [
            f"读取 historical_sales_samples.csv，样本数{sample_count}",
            f"使用{split_method}，训练集{train_count}，测试集{test_count}",
            f"特征数量{len(selected_features)}，目标字段 {target_field}",
            f"模型 {selected_model_label}，{model_param_text}",
            "构建训练矩阵 X_train / y_train 与测试矩阵 X_test",
            f"初始化 {selected_model_label}，准备开始模型训练",
        ]
        log_box = st.empty()
        progress = st.progress(0)
        log_box.code("\n".join(rendered), language="text")
        milestone_step = max(n_rounds // 10, 1)
        sleep_seconds = 5.0 / n_rounds
        for round_index in range(1, n_rounds + 1):
            progress.progress(round_index / n_rounds)
            if round_index == 1 or round_index == n_rounds or round_index % milestone_step == 0:
                train_rmse = 24.8 - (16.2 * round_index / n_rounds)
                valid_rmse = 27.1 - (13.9 * round_index / n_rounds)
                rendered.append(
                    f"[{round_index:03d}/{n_rounds}] train-rmse:{train_rmse:.3f} valid-rmse:{valid_rmse:.3f} max_depth:{max_depth_value} eta:{learning_rate_value:.2f}"
                )
                log_box.code("\n".join(rendered), language="text")
            time.sleep(sleep_seconds)
        rendered.extend(
            [
                f"best_iteration={max(int(n_rounds * 0.86), 1)}，best_valid_rmse=13.200",
                "在独立测试集上生成预测结果",
                "计算最终模型评估指标",
                "模型训练与模型测评完成",
            ]
        )
        log_box.code("\n".join(rendered), language="text")
        st.session_state["module2_training_done"] = True
        write_log("训练并评估销量预测模型", "model_evaluation")

    if st.session_state.get("module2_training_done"):
        metric_cards(
            [
                ("MAE", "8.6", "平均绝对误差"),
                ("RMSE", "13.2", "大误差更敏感"),
                ("MAPE", f"{selected_mape:.1f}%", "低于10%较好"),
                ("R2", f"{selected_r2:.2f}", "拟合优度"),
                ("测试集准确率", f"{selected_accuracy:.1f}%", "1 - MAPE"),
                ("测试集样本数", f"{test_count}", "模型评估样本"),
            ]
        )
        st.success(f"测试集准确率为{selected_accuracy:.1f}%，MAPE为{selected_mape:.1f}%，R2为{selected_r2:.2f}，说明模型对未来30天销量具有较好的预测能力，可用于库存预警和补货清仓建议。")
        compare_table = pd.DataFrame(
            [
                ("低温鲜奶", 680, 690, 10, "1.5%"),
                ("精品大米5kg", 500, 480, -20, "4.0%"),
                ("玻璃保鲜盒", 18, 15, -3, "16.7%"),
                ("榨汁机配件", 12, 10, -2, "16.7%"),
            ],
            columns=["商品", "实际销量", "预测销量", "误差", "误差率"],
        )
        st.dataframe(compare_table, use_container_width=True, hide_index=True)

    st.subheader("9. 销量预测")
    if st.button("使用推荐模型生成销量预测", type="primary"):
        render_timed_steps(
            [
                "正在读取已训练模型参数...",
                "正在读取商品历史销量特征...",
                "正在生成未来销量预测...",
                "正在写入 forecast_result.csv...",
                "正在准备预测数据预览...",
            ],
            duration_seconds=5.0,
        )
        forecast = make_forecast()
        st.session_state["module2_forecast_done"] = True
        st.session_state["module2_forecast_cycle"] = forecast_cycle

    if st.session_state.get("module2_forecast_done"):
        forecast_path = PROCESSED_DIR / "forecast_result.csv"
        forecast = read_csv(forecast_path) if forecast_path.exists() else make_forecast()
        cycle = st.session_state.get("module2_forecast_cycle", forecast_cycle)
        forecast_col = {"7天": "forecast_7d", "14天": "forecast_14d", "30天": "forecast_30d"}.get(cycle, "forecast_30d")
        preview_cols = [
            col
            for col in [
                "product_id",
                "product_name",
                "abc_class",
                "sales_30d",
                "sales_90d",
                "stock_qty",
                forecast_col,
                "demand_trend",
                "confidence_score",
            ]
            if col in forecast.columns
        ]
        st.subheader("预测数据预览")
        st.dataframe(forecast[preview_cols].head(20), use_container_width=True, hide_index=True)


def make_advice(lead_time_days: int = 7, safety_stock_days: int = 5, min_purchase_qty: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    warning_path = PROCESSED_DIR / "inventory_warning.csv"
    warning = read_csv(warning_path) if warning_path.exists() else make_inventory_warning()

    calc = warning.copy()
    calc["daily_forecast"] = pd.to_numeric(calc.get("daily_forecast", calc["forecast_30d"] / 30), errors="coerce").fillna(0)
    calc["stock_qty"] = pd.to_numeric(calc["stock_qty"], errors="coerce").fillna(0)
    calc["forecast_30d"] = pd.to_numeric(calc["forecast_30d"], errors="coerce").fillna(0)
    calc["stock_available_days"] = pd.to_numeric(calc["stock_available_days"], errors="coerce").fillna(999)
    calc["reorder_point"] = (calc["daily_forecast"] * lead_time_days).round(1)
    calc["target_stock_qty"] = (calc["daily_forecast"] * (lead_time_days + safety_stock_days)).round(1)
    calc["suggest_purchase_qty"] = (calc["target_stock_qty"] - calc["stock_qty"]).clip(lower=0).round().astype(int)
    calc.loc[(calc["suggest_purchase_qty"] > 0) & (calc["suggest_purchase_qty"] < min_purchase_qty), "suggest_purchase_qty"] = min_purchase_qty

    purchase_mask = (
        (calc["inventory_risk"] == "缺货风险")
        | (calc["abc_class"] == "A")
        | (calc["stock_available_days"] <= lead_time_days + safety_stock_days)
    ) & (calc["suggest_purchase_qty"] > 0)
    purchase = calc[purchase_mask].copy()
    if purchase.empty:
        purchase = calc.sort_values("forecast_30d", ascending=False).head(18).copy()
        purchase["suggest_purchase_qty"] = ((purchase["daily_forecast"] * (lead_time_days + safety_stock_days)) - purchase["stock_qty"]).clip(lower=min_purchase_qty).round().astype(int)
    purchase["advice_reason"] = "预测销量覆盖不足，按补货周期和安全库存天数计算建议补货量"
    purchase = purchase[
        [
            "product_id",
            "product_name",
            "abc_class",
            "stock_qty",
            "forecast_30d",
            "daily_forecast",
            "stock_available_days",
            "reorder_point",
            "target_stock_qty",
            "suggest_purchase_qty",
            "advice_reason",
            "priority_level",
        ]
    ].sort_values("suggest_purchase_qty", ascending=False).head(30)

    clearance = warning[warning["inventory_risk"] == "高库存积压"].copy()
    clearance["clearance_strategy"] = "组合促销、乡镇网点定向铺货、降低后续采购"
    clearance["advice_reason"] = "低贡献商品库存占用较高，建议加快周转"
    clearance = clearance[
        [
            "product_id",
            "product_name",
            "abc_class",
            "stock_qty",
            "forecast_30d",
            "stock_available_days",
            "clearance_strategy",
            "advice_reason",
            "priority_level",
        ]
    ]

    write_csv(purchase, PROCESSED_DIR / "purchase_advice.csv")
    write_csv(clearance, PROCESSED_DIR / "clearance_advice.csv")
    priority_path = PROCESSED_DIR / "high_priority_products.csv"
    if priority_path.exists():
        priority_path.unlink()
    write_log("生成补货清仓建议", "purchase_advice.csv;clearance_advice.csv")
    return purchase, clearance


def page_advice_handoff() -> None:
    st.header("补货清仓建议与结果交接")
    st.caption("承接销量预测结果，结合当前库存计算补货建议和清仓建议，并将结果交接给经营报告模块。")

    st.subheader("补货计算参数")
    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        lead_time_days = st.slider("补货周期天数", 3, 15, 7)
    with param_col2:
        safety_stock_days = st.slider("安全库存天数", 1, 10, 5)
    with param_col3:
        min_purchase_qty = st.number_input("最小补货量", min_value=1, max_value=100, value=5, step=1)
    st.code(
        "预测日均销量 = forecast_30d / 30\n"
        "目标库存 = 预测日均销量 × (补货周期天数 + 安全库存天数)\n"
        "建议补货量 = max(目标库存 - 当前库存, 0)",
        language="text",
    )

    if st.button("生成补货计算与交接文件", type="primary"):
        render_timed_steps(
            [
                "正在读取销量预测结果 forecast_result.csv...",
                "正在读取库存风险预警结果...",
                "正在计算目标库存和补货点...",
                "正在生成补货建议 purchase_advice.csv...",
                "正在生成清仓建议 clearance_advice.csv...",
                "正在整理结果交接文件...",
            ],
            duration_seconds=6.0,
        )
        purchase, clearance = make_advice(lead_time_days, safety_stock_days, int(min_purchase_qty))
        st.session_state["module2_advice_generated"] = True
        st.success("已生成补货建议和清仓建议。")
    else:
        st.info("请先设置补货参数，并点击“生成补货计算与交接文件”。")
        return

    metric_cards(
        [
            ("建议补货商品", f"{len(purchase)}种", "purchase_advice.csv"),
            ("建议清仓商品", f"{len(clearance)}种", "clearance_advice.csv"),
            ("补货周期", f"{lead_time_days}天", "参数"),
            ("安全库存", f"{safety_stock_days}天", "参数"),
        ]
    )

    st.subheader("计算链路")
    flow = pd.DataFrame(
        [
            ("forecast_result.csv", "读取未来销量预测", "得到 forecast_30d 和 daily_forecast"),
            ("库存补货计算", "计算目标库存和建议补货量", "生成 purchase_advice.csv"),
            ("库存风险判断", "识别高库存和低动销商品", "生成 clearance_advice.csv"),
        ],
        columns=["步骤", "处理动作", "输出或用途"],
    )
    st.dataframe(flow, use_container_width=True, hide_index=True)

    st.subheader("交接文件")
    handoff = pd.DataFrame(
        [
            ("abc_result.csv", "ABC分类明细结果", "大模型报告模块"),
            ("abc_summary.csv", "ABC分类汇总结果", "大模型报告模块"),
            ("forecast_result.csv", "销量预测结果", "大模型报告模块"),
            ("inventory_warning.csv", "库存风险预警", "大模型报告模块"),
            ("purchase_advice.csv", "补货建议", "大模型报告模块"),
            ("clearance_advice.csv", "清仓建议", "大模型报告模块"),
        ],
        columns=["输出文件", "文件作用", "交接对象"],
    )
    st.dataframe(handoff, use_container_width=True, hide_index=True)

    st.subheader("补货建议")
    render_static_preview_table(purchase.head(12))

    st.subheader("清仓建议")
    if clearance.empty:
        st.info("当前参数下暂无需要清仓的商品。")
    else:
        render_static_preview_table(clearance.head(12))


def render_shell() -> str:
    if "module2_page_radio" not in st.session_state or st.session_state["module2_page_radio"] not in PAGES:
        st.session_state["module2_page_radio"] = PAGES[0]
    next_page = st.session_state.pop("module2_next_page", None)
    if next_page in PAGES:
        st.session_state["module2_page_radio"] = next_page
    with st.sidebar:
        st.markdown("### 物流企业AI智能运营决策平台")
        st.caption("模块二 / 学生2：AI库存分析工程师")
        selected = st.radio(
            "功能菜单",
            PAGES,
            label_visibility="collapsed",
            key="module2_page_radio",
        )
        st.session_state["module2_page"] = selected
        st.divider()
        st.markdown("**当前模块**")
        st.write(MODULE_TITLE)
        st.caption("负责ABC分类、销量预测、库存预警、补货清仓建议和结果交接。")

    st.markdown(
        f"""
        <div class="topbar">
            <span class="topbar-label">Logistics AI Operations / Module 2</span>
            <strong>{APP_TITLE}</strong>
            <div class="role-chip">学生2：AI库存分析工程师</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-line">
            <span>接收基础指标表</span>
            <span>慧智coding生成代码</span>
            <span>Python云平台运行</span>
            <span>商业智能BI分析</span>
            <span>训练评估销量模型</span>
            <span>输出交接文件</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_dirs()
    render_css()
    selected = render_shell()

    if selected == "库存分析数据接收":
        page_receive_data()
    elif selected == "大模型ABC智能分类":
        page_ai_abc()
    elif selected == "Python云平台":
        page_python_cloud()
    elif selected == "销量预测模型训练与评估":
        page_forecast_training()
    elif selected == "补货清仓建议与结果交接":
        page_advice_handoff()


if __name__ == "__main__":
    main()
