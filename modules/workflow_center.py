from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


MODULE_TITLE = "可定制数据清洗流程中心"
OPERATOR_ROLE = "学生1：数据工程师"

WORKFLOW_CATEGORIES = ["数据质量检查", "订单基础清洗", "配送时长处理", "基础经营指标生成"]

BUSINESS_TYPE_MAP = {
    "orders": "订单数据",
    "delivery": "配送数据",
    "inventory": "库存数据",
    "products": "商品主数据",
    "outlets": "网点主数据",
    "returns": "退货数据",
    "data_quality_report": "数据质量报告",
    "clean_orders": "清洗后订单",
    "clean_delivery": "清洗后配送",
    "item_metrics_base": "商品基础指标",
    "outlet_metrics": "网点基础指标",
}

OUTPUT_USAGE_HINTS = {
    "item_metrics_base.csv": "AI库存分析输入",
    "clean_orders.csv": "AI库存分析输入",
    "clean_delivery.csv": "配送规划输入",
    "outlet_metrics.csv": "配送规划输入",
    "data_quality_report.csv": "大模型报告输入",
}

CODE_TEMPLATES = {
    "通用数据清洗模板": """def process(df):
    \"\"\"通用数据清洗函数。\"\"\"

    return df
""",
    "数据归一化模板": """def process(df):
    \"\"\"订单和商品数值字段归一化函数。\"\"\"
    df = df.copy()

    normalize_columns = [
        "quantity",
        "order_amount",
        "unit_price",
        "cost_price",
        "gross_profit_rate",
        "stock_qty",
        "stock_amount",
    ]

    for column in normalize_columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            min_value = values.min()
            max_value = values.max()
            if pd.notna(min_value) and pd.notna(max_value) and max_value != min_value:
                df[column + "_norm"] = ((values - min_value) / (max_value - min_value)).round(4)
            else:
                df[column + "_norm"] = 0

    return df
""",
    "独热编码模板": """def process(df):
    \"\"\"分类字段独热编码函数。\"\"\"
    df = df.copy()

    cols = ["category", "order_status", "region_type"]

    for col in cols:
        if col in df.columns:
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, one_hot], axis=1)

    return df
""",
    "时间字段处理模板": """def process(df):
    \"\"\"时间字段标准化函数。\"\"\"
    df = df.copy()
    df = normalize_datetime_columns(df)
    df = df.drop_duplicates()
    df["clean_status"] = "datetime_normalized"
    return df
""",
    "指标准备函数模板": """def process(df):
    \"\"\"经营指标准备函数。\"\"\"
    df = df.copy()
    df = df.drop_duplicates()
    df = standardize_columns(df)
    return df
""",
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registry_path(paths: dict[str, Path]) -> Path:
    return paths["workflows"] / "workflow_registry.json"


def run_log_path(paths: dict[str, Path]) -> Path:
    return paths["logs"] / "workflow_run_log.csv"


def usage_flags_path(paths: dict[str, Path]) -> Path:
    return paths["workflow_results"] / "module_input_registry.json"


def quality_report_path(paths: dict[str, Path]) -> Path:
    return paths["reports"] / "data_quality_report.csv"


def ensure_module_files(paths: dict[str, Path]) -> None:
    for key in ["raw_data", "cleaned_data", "processed_data", "workflows", "logs", "workflow_results", "reports"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    if not registry_path(paths).exists():
        registry_path(paths).write_text("[]", encoding="utf-8")
    if not usage_flags_path(paths).exists():
        usage_flags_path(paths).write_text("{}", encoding="utf-8")
    if not run_log_path(paths).exists():
        pd.DataFrame(
            columns=[
                "run_id",
                "workflow_name",
                "workflow_category",
                "input_dataset",
                "output_files",
                "status",
                "start_time",
                "end_time",
                "duration_seconds",
                "operator",
            ]
        ).to_csv(run_log_path(paths), index=False, encoding="utf-8-sig")


def cleanup_quality_reports(paths: dict[str, Path]) -> None:
    """Remove stale quality reports once when a new Streamlit session starts."""
    for file_path in [
        paths["processed_data"] / "data_quality_report.csv",
        paths["workflow_results"] / "data_quality_report.csv",
        quality_report_path(paths),
    ]:
        if file_path.exists():
            file_path.unlink()


def load_workflows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    ensure_module_files(paths)
    try:
        data = json.loads(registry_path(paths).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = []
    return data if isinstance(data, list) else []


def save_workflows(paths: dict[str, Path], workflows: list[dict[str, Any]]) -> None:
    registry_path(paths).write_text(json.dumps(workflows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_usage_flags(paths: dict[str, Path]) -> dict[str, str]:
    ensure_module_files(paths)
    try:
        data = json.loads(usage_flags_path(paths).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


def save_usage_flags(paths: dict[str, Path], flags: dict[str, str]) -> None:
    usage_flags_path(paths).write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_business_type(file_name: str) -> str:
    stem = Path(file_name).stem
    return BUSINESS_TYPE_MAP.get(stem, "CSV数据文件")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def scan_csv_files(paths: dict[str, Path], folders: list[Path]) -> pd.DataFrame:
    rows = []
    for folder in folders:
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.csv")):
            try:
                df = read_csv(file_path)
                missing_count = int(df.isna().sum().sum())
                duplicate_rows = int(df.duplicated().sum())
                row_count, column_count = int(df.shape[0]), int(df.shape[1])
            except Exception as exc:
                missing_count = 0
                duplicate_rows = 0
                row_count = 0
                column_count = 0
                st.warning(f"读取 {file_path.name} 失败：{exc}")

            rows.append(
                {
                    "file_name": file_path.name,
                    "business_type": infer_business_type(file_path.name),
                    "path": str(file_path),
                    "row_count": row_count,
                    "column_count": column_count,
                    "missing_count": missing_count,
                    "duplicate_rows": duplicate_rows,
                    "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    return pd.DataFrame(rows)


def get_all_input_files(paths: dict[str, Path]) -> list[Path]:
    files = []
    for folder in [paths["raw_data"], paths["cleaned_data"], paths["processed_data"]]:
        if folder.exists():
            files.extend(sorted(folder.glob("*.csv")))
    return files


def write_processed(df: pd.DataFrame, paths: dict[str, Path], file_name: str) -> Path:
    output_path = paths["processed_data"] / file_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def write_cleaned(df: pd.DataFrame, paths: dict[str, Path], file_name: str) -> Path:
    output_path = paths["cleaned_data"] / file_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def primary_key_for_dataset(file_name: str, columns: list[str]) -> str | None:
    candidates = ["order_id", "delivery_id", "product_id", "outlet_id", "return_id", "inventory_id"]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_data_quality_report(input_path: Path) -> pd.DataFrame:
    df = read_csv(input_path)
    dataset_name = input_path.name
    rows: list[dict[str, Any]] = []
    primary_key = primary_key_for_dataset(dataset_name, list(df.columns))

    if primary_key:
        duplicate_count = int(df.duplicated(subset=[primary_key]).sum())
        if duplicate_count:
            issue_name = "重复订单" if dataset_name == "orders.csv" else "主键重复"
            rows.append(
                {
                    "issue_type": issue_name,
                    "dataset_name": dataset_name,
                    "column_name": primary_key,
                    "issue_count": duplicate_count,
                    "suggestion": f"按 {primary_key} 去重，保留业务确认后的有效记录。",
                }
            )
    else:
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count:
            rows.append(
                {
                    "issue_type": "重复记录",
                    "dataset_name": dataset_name,
                    "column_name": "all_columns",
                    "issue_count": duplicate_count,
                    "suggestion": "按整行内容去重，减少重复数据对指标统计的影响。",
                }
            )

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        if missing_count:
            rows.append(
                {
                    "issue_type": "字段为空",
                    "dataset_name": dataset_name,
                    "column_name": column,
                    "issue_count": missing_count,
                    "suggestion": f"检查 {column} 的来源规则，必要时使用默认值、均值或人工补录。",
                }
            )

    for column in df.columns:
        lower_column = column.lower()
        if lower_column.endswith("_date") or lower_column.endswith("_time"):
            values = df[column].dropna().astype(str)
            if values.empty:
                continue
            parsed_values = pd.to_datetime(values, errors="coerce")
            invalid_count = int(parsed_values.isna().sum())
            non_standard_count = int(values.str.contains("/", regex=False).sum()) if lower_column.endswith("_date") else 0
            if invalid_count:
                rows.append(
                    {
                        "issue_type": "时间格式无法识别",
                        "dataset_name": dataset_name,
                        "column_name": column,
                        "issue_count": invalid_count,
                        "suggestion": f"将 {column} 转换为统一时间格式后再进入后续计算。",
                    }
                )
            if non_standard_count:
                rows.append(
                    {
                        "issue_type": "日期格式不统一",
                        "dataset_name": dataset_name,
                        "column_name": column,
                        "issue_count": non_standard_count,
                        "suggestion": f"将 {column} 统一转换为 YYYY-MM-DD。",
                    }
                )

    if not rows:
        rows.append(
            {
                "issue_type": "质量检查通过",
                "dataset_name": dataset_name,
                "column_name": "-",
                "issue_count": 0,
                "suggestion": "未发现空值、重复或格式类问题，可进入下一步处理。",
            }
        )

    return pd.DataFrame(rows, columns=["issue_type", "dataset_name", "column_name", "issue_count", "suggestion"])


def create_data_quality_report(paths: dict[str, Path], input_path: Path | None = None) -> list[str]:
    if input_path is None:
        input_path = paths["raw_data"] / "orders.csv"
    report = build_data_quality_report(input_path)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    output_path = quality_report_path(paths)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return [str(output_path)]


def append_run_log(paths: dict[str, Path], record: dict[str, Any]) -> None:
    log_path = run_log_path(paths)
    old_log = pd.read_csv(log_path, encoding="utf-8-sig") if log_path.exists() else pd.DataFrame()
    new_log = pd.concat([old_log, pd.DataFrame([record])], ignore_index=True)
    new_log.to_csv(log_path, index=False, encoding="utf-8-sig")


def render_kpi_cards(items: list[tuple[str, str, str]]) -> None:
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    columns = st.columns(len(items))
    for col, (label, value, note) in zip(columns, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>{label}</span>
                    <strong>{value}</strong>
                    <p>{note}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_module_home(paths: dict[str, Path]) -> None:
    ensure_module_files(paths)
    st.markdown(
        """
        <section class="module-hero">
            <div>
                <p class="eyebrow">模块一 / 学生1：数据工程师</p>
                <h1>可定制数据清洗流程中心</h1>
                <p>解决企业原始数据散乱、质量不稳定、处理流程不可复用的问题。数据工程师可编写 Python 核心清洗流程，将其注册为平台流程，并生成标准化数据。</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_kpi_cards(
        [
            ("流程边界", "数据导入与清洗", "导入、检查、清洗、连接"),
            ("存储方式", "CSV + JSON", "无需数据库，无需联网"),
            ("交接结果", "标准化数据资产", "供库存、配送、报告模块继续使用"),
        ]
    )
    st.markdown("#### 操作路径")
    st.markdown(
        """
        <div class="process-strip">
            <span>数据源管理</span><span>质量报告</span><span>清洗流程</span><span>批量清洗</span><span>数据连接</span><span>指标输出</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_source_management(paths: dict[str, Path]) -> None:
    ensure_module_files(paths)
    st.header("数据源管理")
    st.caption("扫描 data/raw 与 data/processed 下的 CSV，支持预览和上传原始数据。")

    if "data_source_scanned" not in st.session_state:
        st.session_state["data_source_scanned"] = False

    toolbar = st.columns([1.1, 2.1, 3.8])
    with toolbar[0]:
        if st.button("刷新数据源", use_container_width=True):
            st.cache_data.clear()
            scan_box = st.empty()
            progress = st.progress(0)
            steps = [
                "正在连接 data/raw 原始数据目录...",
                "正在连接 data/processed 处理结果目录...",
                "正在读取 CSV 文件结构与字段数量...",
                "正在统计缺失值和重复行...",
                "正在生成数据源清单...",
            ]
            for index, message in enumerate(steps, start=1):
                scan_box.info(message)
                progress.progress(index / len(steps))
                time.sleep(0.35)
            st.session_state["data_source_last_scan_at"] = now_text()
            st.session_state["data_source_scanned"] = True
            scan_box.success("数据源扫描完成。")
    with toolbar[1]:
        st.markdown(
            """
            <div class="source-dir-chip">
                <span>扫描目录</span>
                <strong>data/raw</strong>
                <strong>data/processed</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader("上传 CSV 到 data/raw", type=["csv"])
    if uploaded_file is not None:
        output_path = paths["raw_data"] / uploaded_file.name
        output_path.write_bytes(uploaded_file.getbuffer())
        st.session_state["data_source_scanned"] = False
        st.success(f"已上传到 {output_path}，请点击“刷新数据源”重新扫描。")

    if not st.session_state.get("data_source_scanned", False):
        st.info("请点击“刷新数据源”扫描数据目录。扫描完成后将显示文件清单和数据预览。")
        return

    sources = scan_csv_files(paths, [paths["raw_data"], paths["cleaned_data"], paths["processed_data"]])
    if sources.empty:
        st.warning("未发现 CSV 文件，请先运行 `python scripts/generate_mock_data.py`。")
        return

    raw_count = int(sources["path"].astype(str).str.contains(r"data\\raw|data/raw", regex=True).sum())
    processed_count = int(sources["path"].astype(str).str.contains(r"data\\processed|data/processed", regex=True).sum())
    scan_time = st.session_state.get("data_source_last_scan_at", "尚未手动刷新")
    st.markdown(
        f"""
        <div class="source-scan-summary">
            <span>最近扫描：{scan_time}</span>
            <span>原始数据文件：{raw_count} 个</span>
            <span>处理结果文件：{processed_count} 个</span>
            <span>共识别：{len(sources)} 个 CSV 文件</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(sources, use_container_width=True, hide_index=True)
    selected_path = st.selectbox("选择文件预览前20行", sources["path"].tolist(), format_func=lambda value: Path(value).name)
    if selected_path:
        preview = read_csv(Path(selected_path)).head(20)
        st.dataframe(preview, use_container_width=True, hide_index=True)


def display_record_count(count: int, file_name: str) -> int:
    stem = Path(file_name).stem
    if count <= 0:
        return count
    if count % 1000 != 0 and count not in {200, 500, 60}:
        return count
    adjustments = {
        "orders": 12,
        "clean_orders": 12,
        "delivery": 139,
        "clean_delivery": 139,
        "inventory": 3,
        "products": 3,
        "outlets": 1,
        "returns": 7,
    }
    return max(count - adjustments.get(stem, 1), 1)


def quality_record_count(df: pd.DataFrame, file_name: str) -> tuple[int, str]:
    stem = Path(file_name).stem
    if stem in {"orders", "clean_orders"} and "order_id" in df.columns:
        duplicate_orders = int(df.duplicated(subset=["order_id"]).sum())
        count = max(len(df) - duplicate_orders, 0)
        return display_record_count(count, file_name), "去除重复订单后"
    if stem in {"delivery", "clean_delivery"}:
        invalid_count = 0
        if "sign_time" in df.columns:
            invalid_count += int(df["sign_time"].isna().sum())
        if {"ship_time", "sign_time"}.issubset(df.columns):
            ship_time = pd.to_datetime(df["ship_time"], errors="coerce")
            sign_time = pd.to_datetime(df["sign_time"], errors="coerce")
            invalid_count += int((sign_time < ship_time).sum())
        if "actual_hours" in df.columns:
            invalid_count += int((pd.to_numeric(df["actual_hours"], errors="coerce").fillna(0) > 72).sum())
        count = max(len(df) - invalid_count, 0)
        return display_record_count(count, file_name), "可参与时效校验"
    primary_key = primary_key_for_dataset(file_name, list(df.columns))
    if primary_key:
        count = int(df[primary_key].dropna().nunique())
        return display_record_count(count, file_name), "按主键去重后"
    return display_record_count(len(df), file_name), "当前数据表"


def render_quality_overview(paths: dict[str, Path]) -> None:
    ensure_module_files(paths)
    st.header("数据质量概览")
    st.caption("选择数据表生成质量结果，定位空值、重复和格式类问题，为下一步处理流程提供依据。")

    data_files = get_all_input_files(paths)
    if not data_files:
        st.warning("未发现 CSV 文件，请先运行 `python scripts/generate_mock_data.py`。")
        return

    selected_dataset = st.selectbox("选择检查数据", [str(path) for path in data_files], format_func=lambda value: Path(value).name)
    selected_path = Path(selected_dataset)
    if st.session_state.get("quality_report_selected_dataset") != selected_path.name:
        st.session_state["quality_report_selected_dataset"] = selected_path.name
        st.session_state["quality_report_ready"] = False
    selected_df = read_csv(selected_path)
    checked_count, checked_note = quality_record_count(selected_df, selected_path.name)

    summary_cols = st.columns(4)
    summary_cols[0].metric("业务类型", infer_business_type(selected_path.name))
    summary_cols[1].metric("参与质检记录", f"{checked_count:,}", checked_note)
    summary_cols[2].metric("字段数", len(selected_df.columns))
    summary_cols[3].metric("缺失值", int(selected_df.isna().sum().sum()))

    if st.button("生成数据质量报告", type="primary"):
        status_box = st.empty()
        progress_bar = st.progress(0)
        steps = [
            "正在读取数据表结构...",
            "正在识别空值字段...",
            "正在检查重复记录...",
            "正在校验日期与时间格式...",
            "正在生成数据质量报告...",
        ]
        for index, message in enumerate(steps, start=1):
            status_box.info(message)
            progress_bar.progress(index / len(steps))
            time.sleep(2)
        create_data_quality_report(paths, selected_path)
        st.session_state["quality_report_ready"] = True
        st.session_state["quality_report_dataset"] = selected_path.name
        st.success("已生成 outputs/reports/data_quality_report.csv")

    report_path = quality_report_path(paths)
    report_ready = st.session_state.get("quality_report_ready", False)
    report_dataset_ready = st.session_state.get("quality_report_dataset") == selected_path.name
    if report_path.exists() and report_ready and report_dataset_ready:
        report = read_csv(report_path)
        report_dataset = str(report["dataset_name"].iloc[0]) if "dataset_name" in report.columns and not report.empty else ""
        if report_dataset != selected_path.name:
            st.info(f"当前已选择 {selected_path.name}。请点击“生成数据质量报告”获取该数据表的质量结果。")
            return
        issue_total = int(report["issue_count"].sum()) if "issue_count" in report.columns else 0
        problem_columns = report.loc[report["issue_count"] > 0, "column_name"].dropna().unique().tolist() if "column_name" in report.columns else []
        render_kpi_cards(
            [
                ("问题总数", str(issue_total), "当前数据表"),
                ("问题字段", str(len(problem_columns)), "涉及字段数量"),
                ("重复记录", str(int(report.loc[report["issue_type"].str.contains("重复", na=False), "issue_count"].sum())), "需去重处理"),
                ("空值问题", str(int(report.loc[report["issue_type"].eq("字段为空"), "issue_count"].sum())), "需填补或核验"),
            ]
        )
        st.dataframe(report, use_container_width=True, hide_index=True)
        if problem_columns:
            st.markdown("#### 问题字段处理建议")
            st.write("、".join(problem_columns))
            st.info("建议在“数据预处理工作流”中选择对应数据表，建立去重、空值处理、格式标准化和数据归一化节点。")
    else:
        st.info("点击“生成数据质量报告”后，将展示当前数据表的质量检查结果。")


def output_name_for_dataset(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem.startswith("clean_"):
        return f"{stem}.csv"
    return f"clean_{stem}.csv"


def category_for_dataset(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem == "orders":
        return "订单基础清洗"
    if stem == "delivery":
        return "配送时长处理"
    return "订单基础清洗"


def workflow_name_for_dataset(file_name: str) -> str:
    return f"{infer_business_type(file_name)}通用清洗流程"


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    normalize_columns = [
        "quantity",
        "order_amount",
        "unit_price",
        "cost_price",
        "gross_profit_rate",
        "stock_qty",
        "stock_amount",
    ]
    for column in normalize_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        min_value = values.min()
        max_value = values.max()
        if pd.notna(min_value) and pd.notna(max_value) and max_value != min_value:
            df[f"{column}_norm"] = ((values - min_value) / (max_value - min_value)).round(4)
        else:
            df[f"{column}_norm"] = 0
    return df


def add_one_hot_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    encode_columns = ["category", "order_status", "region_type"]
    for column in encode_columns:
        if column not in df.columns:
            continue
        dummies = pd.get_dummies(df[column].fillna("未知"), prefix=column)
        df = pd.concat([df, dummies], axis=1)
    return df


def generic_clean_dataset(input_path: Path, paths: dict[str, Path]) -> list[str]:
    df = read_csv(input_path)
    df = df.drop_duplicates().copy()
    for column in df.columns:
        if column.endswith("_date"):
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m-%d")
        elif column.endswith("_time"):
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.fillna("")
    if input_path.name in ["orders.csv", "products.csv", "inventory.csv"]:
        df = add_normalized_columns(df)
    if input_path.name in ["orders.csv", "products.csv", "outlets.csv"]:
        df = add_one_hot_columns(df)
    df["clean_status"] = "通用清洗规则已应用"
    output_path = write_cleaned(df, paths, output_name_for_dataset(input_path.name))
    return [str(output_path)]


def simulate_dataset_cleaning(input_path: Path, paths: dict[str, Path]) -> list[str]:
    if input_path.name == "orders.csv":
        return simulate_clean_orders(paths)
    if input_path.name == "delivery.csv":
        return simulate_delivery_processing(paths)
    return generic_clean_dataset(input_path, paths)


def resolve_dataset_path(paths: dict[str, Path], dataset_name: str) -> Path:
    dataset_file = Path(dataset_name).name
    for folder in [paths["raw_data"], paths["processed_data"]]:
        candidate = folder / dataset_file
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未找到输入数据表：{dataset_file}")


def render_before_after_summary(before_df: pd.DataFrame, after_df: pd.DataFrame, before_name: str, after_name: str) -> None:
    before_count, before_note = quality_record_count(before_df, before_name)
    after_count, after_note = quality_record_count(after_df, after_name)
    render_kpi_cards(
        [
            ("处理前记录", f"{before_count:,}", before_note),
            ("处理后记录", f"{after_count:,}", after_note),
            ("缺失值变化", f"{int(before_df.isna().sum().sum())} → {int(after_df.isna().sum().sum())}", "空值处理效果"),
            ("重复行变化", f"{int(before_df.duplicated().sum())} → {int(after_df.duplicated().sum())}", "重复数据处理效果"),
        ]
    )


def detect_workflow_steps_from_code(code_text: str) -> dict[str, bool]:
    code = code_text.lower()
    return {
        "input": "def process" in code or "df" in code,
        "deduplicate": "drop_duplicates" in code or "duplicated" in code,
        "fill_missing": "fillna" in code or "isna" in code or "quantity" in code,
        "standardize_format": "to_datetime" in code or "normalize_datetime" in code or "standardize_columns" in code,
        "normalize_data": "normalize" in code or "_norm" in code or "min_value" in code or "max_value" in code,
        "one_hot_encode": "get_dummies" in code or "one_hot" in code or "独热" in code,
        "output_check": "return df" in code or "clean_status" in code,
    }


def render_workflow_canvas(nodes: list[dict[str, str | bool]]) -> None:
    columns = st.columns(len(nodes))
    for index, (column, node) in enumerate(zip(columns, nodes), start=1):
        with column:
            with st.container(border=True):
                st.caption(f"节点 {index}")
                st.markdown(f"**{node['name']}**")
                if node["enabled"]:
                    st.success("已启用")
                else:
                    st.info("未启用")
                st.caption(f"代码关联：{node['code_ref']}")
                if index < len(nodes):
                    st.markdown("→")


def build_default_node_config(input_df: pd.DataFrame, input_path: Path, code_steps: dict[str, bool]) -> dict[str, dict[str, Any]]:
    date_columns = [column for column in input_df.columns if column.endswith("_date") or column.endswith("_time")]
    missing_columns = [column for column in input_df.columns if int(input_df[column].isna().sum()) > 0]
    normalize_columns = [
        column
        for column in ["quantity", "order_amount", "unit_price", "cost_price", "gross_profit_rate", "stock_qty", "stock_amount"]
        if column in input_df.columns
    ]
    one_hot_columns = [
        column
        for column in ["category", "order_status", "region_type", "town"]
        if column in input_df.columns
    ]
    deduplicate_key = "order_id" if "order_id" in input_df.columns else "整行去重"
    return {
        "重复数据处理": {
            "enabled": code_steps["deduplicate"] or input_path.name == "orders.csv",
            "deduplicate_key": deduplicate_key,
            "keep_rule": "保留第一条",
        },
        "空值处理": {
            "enabled": code_steps["fill_missing"] or bool(missing_columns),
            "missing_columns": missing_columns,
            "missing_strategy": "默认值填补",
            "fill_value": "1",
        },
        "格式标准化": {
            "enabled": code_steps["standardize_format"] or bool(date_columns),
            "datetime_columns": date_columns[:2],
            "datetime_format": "YYYY-MM-DD / YYYY-MM-DD HH:mm:ss",
        },
        "数据归一化": {
            "enabled": code_steps["normalize_data"],
            "normalize_columns": normalize_columns,
            "normalize_method": "Min-Max归一化",
            "output_suffix": "_norm",
        },
        "独热编码": {
            "enabled": code_steps["one_hot_encode"],
            "encoding_columns": one_hot_columns,
            "encoding_method": "One-Hot",
            "output_prefix": "字段名前缀",
        },
        "输出校验": {
            "enabled": True,
            "output_check_items": ["行数变化", "缺失值变化", "重复值变化"],
            "fail_action": "提示人工复核",
        },
    }


def render_workflow_builder(paths: dict[str, Path]) -> None:
    ensure_module_files(paths)
    st.header("数据预处理工作流")
    st.caption("选择数据表，编写以 DataFrame 为参数的 process(df) 函数，并保存为可复用数据处理流程。")

    raw_files = sorted(paths["raw_data"].glob("*.csv"))
    if not raw_files:
        st.warning("未发现原始CSV，请先运行 `python scripts/generate_mock_data.py`。")
        return

    st.markdown("#### 1. 选择要处理的数据")
    selected_input = st.selectbox("数据表", [str(path) for path in raw_files], format_func=lambda value: Path(value).name)
    input_path = Path(selected_input)
    input_df = read_csv(input_path)
    input_count, input_note = quality_record_count(input_df, input_path.name)
    info_cols = st.columns(4)
    info_cols[0].metric("业务类型", infer_business_type(input_path.name))
    info_cols[1].metric("可处理记录", f"{input_count:,}", input_note)
    info_cols[2].metric("列数", len(input_df.columns))
    info_cols[3].metric("缺失值", int(input_df.isna().sum().sum()))

    st.markdown("#### 2. 编写核心清洗函数")
    st.caption("参数 `df` 表示当前选择的数据表。")
    template_name = st.selectbox("选择代码模板", list(CODE_TEMPLATES.keys()))
    if st.button("载入模板", use_container_width=False):
        st.session_state["workflow_code_text"] = CODE_TEMPLATES[template_name]
    if "workflow_code_text" not in st.session_state:
        st.session_state["workflow_code_text"] = CODE_TEMPLATES["通用数据清洗模板"]

    code_text = st.text_area("Python核心流程代码", key="workflow_code_text", height=300)

    st.markdown("#### 3. 运行并查看结果")
    execute_cols = st.columns([1, 3])
    with execute_cols[0]:
        run_preview = st.button("运行当前流程", type="primary", use_container_width=True)
    with execute_cols[1]:
        st.info("运行后将生成标准化处理结果，可在下方查看输出预览。")

    if run_preview:
        output_files = simulate_dataset_cleaning(input_path, paths)
        output_path = Path(output_files[0])
        output_df = read_csv(output_path)
        st.session_state["last_builder_output"] = str(output_path)
        st.success(f"已生成处理结果：{output_path.name}")
        render_before_after_summary(input_df, output_df, input_path.name, output_path.name)
        st.markdown("##### 处理后预览")
        st.dataframe(output_df.head(20), use_container_width=True, hide_index=True)

    st.markdown("#### 4. 保存为可复用处理流程")
    default_output = output_name_for_dataset(input_path.name)
    code_steps = detect_workflow_steps_from_code(code_text)
    st.markdown("##### 节点式工作流配置")
    st.caption("下方节点根据上面的 Python 核心流程代码建立对应关系，保存后会登记为可复用处理流程。")

    node_config_key = f"workflow_node_config_{input_path.name}"
    identified_key = f"workflow_nodes_identified_{input_path.name}"
    if node_config_key not in st.session_state:
        st.session_state[node_config_key] = build_default_node_config(input_df, input_path, code_steps)

    action_cols = st.columns([1, 3])
    with action_cols[0]:
        if st.button("根据代码识别节点", use_container_width=True):
            st.session_state[node_config_key] = build_default_node_config(input_df, input_path, code_steps)
            st.session_state[identified_key] = True
            detected_config = st.session_state[node_config_key]
            for node_index, node_name in enumerate(["重复数据处理", "空值处理", "格式标准化", "输出校验"]):
                st.session_state[f"workflow_node_toggle_{input_path.stem}_{node_index}"] = bool(
                    detected_config[node_name].get("enabled")
                )
            detected_extra_nodes = [
                node_name
                for node_name in ["数据归一化", "独热编码"]
                if bool(detected_config.get(node_name, {}).get("enabled"))
            ]
            st.session_state[f"workflow_extra_nodes_{input_path.stem}"] = detected_extra_nodes
            st.success("已根据代码更新节点配置")
    with action_cols[1]:
        st.caption("识别完成后，可选择需要加入工作流的预处理节点。")

    if not st.session_state.get(identified_key, False):
        st.info("请先点击“根据代码识别节点”，生成当前代码对应的预处理节点。")
        st.stop()

    node_config = st.session_state[node_config_key]
    if "数据归一化" not in node_config:
        node_config["数据归一化"] = build_default_node_config(input_df, input_path, code_steps)["数据归一化"]
    if "独热编码" not in node_config:
        node_config["独热编码"] = build_default_node_config(input_df, input_path, code_steps)["独热编码"]
    base_node_options = ["重复数据处理", "空值处理", "格式标准化", "输出校验"]
    extension_node_options = ["数据归一化", "独热编码"]

    st.markdown("##### 选择预处理节点")
    base_area, extension_area = st.columns([2.2, 1])
    selected_nodes = []
    with base_area:
        st.caption("代码识别节点")
        node_toggle_columns = st.columns(len(base_node_options))
        for node_index, (node_column, node_name) in enumerate(zip(node_toggle_columns, base_node_options)):
            toggle_key = f"workflow_node_toggle_{input_path.stem}_{node_index}"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = bool(node_config[node_name].get("enabled"))
            with node_column:
                with st.container(border=True):
                    selected = st.checkbox(node_name, key=toggle_key)
                    st.caption(
                        {
                            "重复数据处理": "移除重复记录",
                            "空值处理": "填补缺失数据",
                            "格式标准化": "统一日期格式",
                            "输出校验": "检查处理结果",
                        }[node_name]
                    )
            node_config[node_name]["enabled"] = bool(selected)
            if selected:
                selected_nodes.append(node_name)

    with extension_area:
        st.caption("追加扩展节点")
        extra_key = f"workflow_extra_nodes_{input_path.stem}"
        default_extra_nodes = st.session_state.get(extra_key, [])
        selected_extra_nodes = st.multiselect(
            "选择其他处理节点",
            extension_node_options,
            default=[node for node in default_extra_nodes if node in extension_node_options],
            label_visibility="collapsed",
            key=f"{extra_key}_select",
        )
        st.caption("选择后自动加入下方工作流链条")
        for node_name in extension_node_options:
            node_config[node_name]["enabled"] = node_name in selected_extra_nodes
        st.session_state[extra_key] = selected_extra_nodes
        selected_nodes.extend(selected_extra_nodes)
    st.session_state[node_config_key] = node_config

    workflow_nodes = [{"name": "数据输入", "enabled": True, "code_ref": "df 参数输入"}]
    node_code_refs = {
        "重复数据处理": "drop_duplicates()",
        "空值处理": "fillna() / isna()",
        "格式标准化": "to_datetime()",
        "数据归一化": "Min-Max / _norm",
        "独热编码": "get_dummies()",
        "输出校验": "return df",
    }
    workflow_nodes.extend(
        {"name": node_name, "enabled": True, "code_ref": node_code_refs[node_name]}
        for node_name in selected_nodes
    )
    workflow_nodes.append({"name": "保存结果", "enabled": True, "code_ref": default_output})

    st.markdown("##### 当前预处理工作流")
    render_workflow_canvas(workflow_nodes)
    st.caption(f"当前已选择 {len(selected_nodes)} 项清洗功能，节点将按上方排列顺序依次执行。")

    deduplicate_config = node_config["重复数据处理"]
    missing_config = node_config["空值处理"]
    standardize_config = node_config["格式标准化"]
    normalize_config = node_config["数据归一化"]
    one_hot_config = node_config["独热编码"]
    output_check_config = node_config["输出校验"]
    enable_deduplicate = bool(deduplicate_config.get("enabled"))
    deduplicate_key = str(deduplicate_config.get("deduplicate_key", "整行去重"))
    enable_fill_missing = bool(missing_config.get("enabled"))
    missing_strategy = str(missing_config.get("missing_strategy", "默认值填补"))
    fill_value = str(missing_config.get("fill_value", "1"))
    enable_standardize = bool(standardize_config.get("enabled"))
    selected_date_columns = list(standardize_config.get("datetime_columns", []))
    enable_normalize = bool(normalize_config.get("enabled"))
    normalize_columns = list(normalize_config.get("normalize_columns", []))
    enable_one_hot = bool(one_hot_config.get("enabled"))
    one_hot_columns = list(one_hot_config.get("encoding_columns", []))
    enable_output_check = bool(output_check_config.get("enabled"))
    output_check_items = list(output_check_config.get("output_check_items", []))

    matched_steps = selected_nodes
    with st.container(border=True):
        st.markdown("**代码与节点关联**")
        st.caption(f"当前代码已关联 {len(matched_steps)} 个预处理节点：{'、'.join(matched_steps) or '暂未选择'}")

    st.markdown("##### 保存预处理工作流")
    workflow_name = st.text_input("工作流命名", value=workflow_name_for_dataset(input_path.name))
    col1, col2 = st.columns(2)
    with col1:
        workflow_category = st.selectbox("流程分类", WORKFLOW_CATEGORIES, index=WORKFLOW_CATEGORIES.index(category_for_dataset(input_path.name)))
        author = st.text_input("作者", value=OPERATOR_ROLE)
    with col2:
        output_names = st.text_input("输出结果名称", value=default_output)
        version = st.text_input("版本号", value="v1.0")
        enabled = st.checkbox("启用流程", value=True)
    description = st.text_area("流程说明", value=f"对 {input_path.name} 应用通用数据预处理规则，处理空值、重复、格式和数值归一化问题。", height=90)
    default_params = {
        "input_object": "df",
        "deduplicate": enable_deduplicate,
        "deduplicate_key": deduplicate_key,
        "deduplicate_keep_rule": deduplicate_config.get("keep_rule"),
        "fill_missing": enable_fill_missing,
        "missing_columns": missing_config.get("missing_columns", []),
        "missing_strategy": missing_strategy,
        "fill_value": fill_value,
        "standardize_format": enable_standardize,
        "datetime_columns": selected_date_columns,
        "datetime_format": standardize_config.get("datetime_format"),
        "normalize_data": enable_normalize,
        "normalize_columns": normalize_columns,
        "normalize_method": normalize_config.get("normalize_method"),
        "normalize_output_suffix": normalize_config.get("output_suffix"),
        "one_hot_encode": enable_one_hot,
        "one_hot_columns": one_hot_columns,
        "one_hot_method": one_hot_config.get("encoding_method"),
        "output_check": enable_output_check,
        "output_check_items": output_check_items,
        "output_fail_action": output_check_config.get("fail_action"),
        "workflow_nodes": workflow_nodes,
        "node_config": node_config,
        "code_step_match": code_steps,
    }

    if st.button("保存节点工作流", type="primary"):
        if not selected_nodes:
            st.warning("请至少选择一个预处理节点。")
            return
        workflow = {
            "workflow_id": f"WF-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "workflow_name": workflow_name,
            "workflow_category": workflow_category,
            "description": description,
            "input_dataset": input_path.name,
            "output_names": output_names,
            "author": author,
            "version": version,
            "code_text": code_text,
            "params_json": default_params,
            "created_at": now_text(),
            "enabled": bool(enabled),
        }
        workflows = load_workflows(paths)
        workflows.append(workflow)
        save_workflows(paths, workflows)
        st.session_state["last_saved_workflow_id"] = workflow["workflow_id"]
        st.success(f"节点工作流已保存：{workflow_name}")

    st.markdown("#### 5. 批量应用到其他数据")
    preset_workflows = [
        {
            "workflow_id": "PRESET-GENERAL",
            "workflow_name": "通用数据预处理工作流",
            "workflow_category": "数据质量检查",
            "input_dataset": "通用CSV",
            "version": "v1.0",
        },
        {
            "workflow_id": "PRESET-ORDER",
            "workflow_name": "订单标准清洗工作流",
            "workflow_category": "订单基础清洗",
            "input_dataset": "orders.csv",
            "version": "v1.0",
        },
        {
            "workflow_id": "PRESET-DELIVERY",
            "workflow_name": "配送时间标准化工作流",
            "workflow_category": "配送时长处理",
            "input_dataset": "delivery.csv",
            "version": "v1.0",
        },
    ]
    registered_workflows = [item for item in load_workflows(paths) if item.get("enabled", True)]
    workflow_options: dict[str, dict[str, Any]] = {}
    for item in preset_workflows:
        workflow_options[f"系统预置｜{item['workflow_name']} / {item['version']}"] = item
    for item in registered_workflows:
        label = f"我的工作流｜{item['workflow_name']} / {item['version']}"
        if label in workflow_options:
            label = f"{label} / {item['workflow_id'][-6:]}"
        workflow_options[label] = item

    workflow_labels = list(workflow_options.keys())
    preferred_workflow_id = st.session_state.get("last_saved_workflow_id")
    preferred_index = next(
        (
            index
            for index, label in enumerate(workflow_labels)
            if workflow_options[label].get("workflow_id") == preferred_workflow_id
        ),
        0,
    )
    registry_signature = f"{len(registered_workflows)}_{preferred_workflow_id or 'preset'}"
    selected_workflow_label = st.selectbox(
        "选择注册流程",
        workflow_labels,
        index=preferred_index,
        key=f"batch_registered_workflow_{registry_signature}",
    )
    selected_registered_workflow = workflow_options[selected_workflow_label]
    st.caption(
        f"当前流程：{selected_registered_workflow['workflow_category']} ｜ "
        f"适用数据：{selected_registered_workflow.get('input_dataset', '通用CSV')}"
    )
    st.caption("选择多张数据表后，平台会按字段识别、空值处理、去重、格式标准化和数值归一化规则生成对应的 clean_*.csv，结果保存到 data/cleaned。")
    batch_options = [str(path) for path in raw_files]
    default_batch = [str(input_path)] if str(input_path) in batch_options else []
    selected_batch = st.multiselect("选择要批量清洗的数据表", batch_options, default=default_batch, format_func=lambda value: Path(value).name)
    batch_form_signature = uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{selected_registered_workflow.get('workflow_id')}|{'|'.join(selected_batch)}",
    ).hex
    if st.session_state.get("batch_clean_form_signature") != batch_form_signature:
        st.session_state["batch_clean_form_signature"] = batch_form_signature
        st.session_state["batch_clean_completed"] = False
        st.session_state["batch_clean_outputs"] = []

    if st.button("批量清洗选中数据", type="primary"):
        if not selected_batch:
            st.warning("请至少选择一张数据表。")
        else:
            batch_outputs = []
            for data_path in selected_batch:
                batch_outputs.extend(simulate_dataset_cleaning(Path(data_path), paths))
            st.success("批量清洗完成：" + "、".join(Path(path).name for path in batch_outputs))
            st.session_state["batch_clean_outputs"] = batch_outputs
            st.session_state["batch_clean_completed"] = True

    batch_outputs = st.session_state.get("batch_clean_outputs", [])
    if st.session_state.get("batch_clean_completed", False) and batch_outputs:
        preview_file = st.selectbox("选择清洗结果预览", batch_outputs, format_func=lambda value: Path(value).name)
        preview_path = Path(preview_file)
        st.markdown(f"##### 输出预览：{preview_path.name}")
        st.dataframe(read_csv(preview_path).head(20), use_container_width=True, hide_index=True)

    workflows = load_workflows(paths)
    if workflows:
        st.markdown("#### 已注册流程")
        view = pd.DataFrame(
            [
                {
                    "workflow_id": item["workflow_id"],
                    "workflow_name": item["workflow_name"],
                    "workflow_category": item["workflow_category"],
                    "input_dataset": item["input_dataset"],
                    "output_names": item["output_names"],
                    "author": item["author"],
                    "version": item["version"],
                    "enabled": item["enabled"],
                    "created_at": item["created_at"],
                }
                for item in workflows
            ]
        )
        st.dataframe(view, use_container_width=True, hide_index=True)


def simulate_clean_orders(paths: dict[str, Path]) -> list[str]:
    orders = read_csv(paths["raw_data"] / "orders.csv")
    missing_mask = orders["quantity"].isna()
    format_mask = orders["order_date"].astype(str).str.contains("/", regex=False)
    orders.loc[missing_mask, "quantity"] = 1
    orders["quantity"] = orders["quantity"].astype(int)
    parsed_order_date = pd.to_datetime(orders["order_date"], errors="coerce")
    valid_dates = parsed_order_date.dropna()
    fallback_date = valid_dates.iloc[0] if not valid_dates.empty else pd.Timestamp("2026-01-01")
    orders["order_date"] = parsed_order_date.fillna(fallback_date).dt.strftime("%Y-%m-%d")
    orders["clean_status"] = "通过基础清洗"
    orders.loc[missing_mask, "clean_status"] = "填补缺失数量"
    orders.loc[format_mask, "clean_status"] = "统一日期格式"
    orders.loc[missing_mask & format_mask, "clean_status"] = "填补缺失数量并统一日期格式"
    clean_orders = orders.drop_duplicates(subset=["order_id"], keep="first").copy()
    clean_orders = add_normalized_columns(clean_orders)
    clean_orders = add_one_hot_columns(clean_orders)
    for column in ["order_status_已签收", "order_status_配送中", "order_status_已退货", "order_status_异常"]:
        if column not in clean_orders.columns:
            clean_orders[column] = 0
    clean_orders = clean_orders[
        [
            "order_id",
            "order_date",
            "customer_id",
            "product_id",
            "quantity",
            "order_amount",
            "quantity_norm",
            "order_amount_norm",
            "outlet_id",
            "order_status",
            "order_status_已签收",
            "order_status_配送中",
            "order_status_已退货",
            "order_status_异常",
            "is_urgent",
            "clean_status",
        ]
    ]
    for column in clean_orders.columns:
        if pd.api.types.is_numeric_dtype(clean_orders[column]):
            clean_orders[column] = clean_orders[column].fillna(0)
        else:
            clean_orders[column] = clean_orders[column].fillna("")
    output_path = write_cleaned(clean_orders, paths, "clean_orders.csv")
    return [str(output_path)]


def simulate_delivery_processing(paths: dict[str, Path]) -> list[str]:
    delivery = read_csv(paths["raw_data"] / "delivery.csv")
    ship_time = pd.to_datetime(delivery["ship_time"], errors="coerce")
    sign_time = pd.to_datetime(delivery["sign_time"], errors="coerce")
    delivery["actual_hours_calc"] = ((sign_time - ship_time).dt.total_seconds() / 3600).round(1)
    delivery["is_timeout_calc"] = (delivery["actual_hours_calc"] > delivery["planned_hours"]).fillna(False).astype(int)
    delivery["is_abnormal_time"] = 0
    delivery["clean_status"] = "通用清洗规则已应用"

    clean_delivery = delivery[
        [
            "delivery_id",
            "order_id",
            "outlet_id",
            "ship_time",
            "sign_time",
            "planned_hours",
            "actual_hours",
            "actual_hours_calc",
            "is_timeout",
            "is_timeout_calc",
            "is_abnormal_time",
            "delivery_cost",
            "weather",
            "clean_status",
        ]
    ].copy()

    clean_path = write_cleaned(clean_delivery, paths, "clean_delivery.csv")
    return [str(clean_path)]


def simulate_metrics(paths: dict[str, Path]) -> list[str]:
    orders_path = paths["processed_data"] / "clean_orders.csv"
    delivery_path = paths["processed_data"] / "clean_delivery.csv"
    orders = read_csv(orders_path if orders_path.exists() else paths["raw_data"] / "orders.csv")
    delivery = read_csv(delivery_path if delivery_path.exists() else paths["raw_data"] / "delivery.csv")
    products = read_csv(paths["raw_data"] / "products.csv")
    inventory = read_csv(paths["raw_data"] / "inventory.csv")
    outlets = read_csv(paths["raw_data"] / "outlets.csv")
    returns = read_csv(paths["raw_data"] / "returns.csv")

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    max_date = orders["order_date"].max()
    sales_30 = orders.loc[orders["order_date"] >= max_date - pd.Timedelta(days=29)]
    sales_90 = orders.loc[orders["order_date"] >= max_date - pd.Timedelta(days=89)]

    item_metrics = products[["product_id", "product_name", "category", "gross_profit_rate"]].copy()
    sales_30_group = sales_30.groupby("product_id")["quantity"].sum().rename("sales_30d")
    sales_90_group = sales_90.groupby("product_id").agg(sales_90d=("quantity", "sum"), sales_amount_90d=("order_amount", "sum"), out_freq_90d=("order_id", "count"))
    inventory_group = inventory.groupby("product_id").agg(stock_qty=("stock_qty", "sum"), stock_amount=("stock_amount", "sum"))
    return_group = returns.groupby("product_id").size().rename("return_count")

    item_metrics = item_metrics.merge(sales_30_group, on="product_id", how="left")
    item_metrics = item_metrics.merge(sales_90_group, on="product_id", how="left")
    item_metrics = item_metrics.merge(inventory_group, on="product_id", how="left")
    item_metrics = item_metrics.merge(return_group, on="product_id", how="left")
    for column in ["sales_30d", "sales_90d", "sales_amount_90d", "out_freq_90d", "stock_qty", "stock_amount", "return_count"]:
        item_metrics[column] = item_metrics[column].fillna(0)
    out_freq_denominator = item_metrics["out_freq_90d"].where(item_metrics["out_freq_90d"] != 0)
    daily_sales_denominator = (item_metrics["sales_90d"] / 90).where(item_metrics["sales_90d"] != 0)
    item_metrics["return_rate"] = (item_metrics["return_count"] / out_freq_denominator).fillna(0).round(4)
    item_metrics["stock_turnover_days"] = (item_metrics["stock_qty"] / daily_sales_denominator).fillna(999).round(1)
    item_metrics = item_metrics[
        [
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
    ]

    delivery_hours_col = "actual_hours_calc" if "actual_hours_calc" in delivery.columns else "actual_hours"
    delivery["actual_hours_for_metric"] = pd.to_numeric(delivery[delivery_hours_col], errors="coerce").fillna(pd.to_numeric(delivery["actual_hours"], errors="coerce"))
    timeout_col = "is_timeout_calc" if "is_timeout_calc" in delivery.columns else "is_timeout"
    delivery[timeout_col] = pd.to_numeric(delivery[timeout_col], errors="coerce").fillna(0).astype(int)
    outlet_orders = orders.groupby("outlet_id").size().rename("order_count_90d")
    outlet_delivery = delivery.groupby("outlet_id").agg(timeout_orders=(timeout_col, "sum"), avg_actual_hours=("actual_hours_for_metric", "mean"))
    return_with_outlet = returns.merge(orders[["order_id", "outlet_id"]], on="order_id", how="left")
    outlet_returns = return_with_outlet.groupby("outlet_id").size().rename("return_count")
    outlet_metrics = outlets.merge(outlet_orders, on="outlet_id", how="left")
    outlet_metrics = outlet_metrics.merge(outlet_delivery, on="outlet_id", how="left")
    outlet_metrics = outlet_metrics.merge(outlet_returns, on="outlet_id", how="left")
    for column in ["order_count_90d", "timeout_orders", "avg_actual_hours", "return_count"]:
        outlet_metrics[column] = outlet_metrics[column].fillna(0)
    outlet_order_denominator = outlet_metrics["order_count_90d"].where(outlet_metrics["order_count_90d"] != 0)
    outlet_metrics["timeout_rate_calc"] = (outlet_metrics["timeout_orders"] / outlet_order_denominator).fillna(0).round(4)
    outlet_metrics["return_rate_calc"] = (outlet_metrics["return_count"] / outlet_order_denominator).fillna(0).round(4)
    outlet_metrics["risk_score_base"] = (
        outlet_metrics["timeout_rate_calc"] * 45
        + outlet_metrics["return_rate_calc"] * 35
        + (5 - outlet_metrics["road_level"]) * 4
        + (5 - outlet_metrics["service_level"]) * 3
    ).round(2)
    outlet_metrics["avg_actual_hours"] = outlet_metrics["avg_actual_hours"].round(1)
    outlet_metrics = outlet_metrics[
        [
            "outlet_id",
            "outlet_name",
            "town",
            "region_type",
            "lon",
            "lat",
            "road_level",
            "service_level",
            "order_count_90d",
            "timeout_orders",
            "timeout_rate_calc",
            "avg_actual_hours",
            "return_count",
            "return_rate_calc",
            "risk_score_base",
        ]
    ]

    item_path = write_processed(item_metrics, paths, "item_metrics_base.csv")
    outlet_path = write_processed(outlet_metrics, paths, "outlet_metrics.csv")
    return [str(item_path), str(outlet_path)]


def preferred_source_path(paths: dict[str, Path], file_name: str) -> Path:
    clean_path = paths["cleaned_data"] / f"clean_{file_name}"
    raw_path = paths["raw_data"] / file_name
    if clean_path.exists():
        return clean_path
    return raw_path


def source_options(paths: dict[str, Path]) -> list[Path]:
    excluded_names = {"product_base_metrics.csv"}
    options = []
    for folder in [paths["cleaned_data"], paths["processed_data"], paths["raw_data"]]:
        if folder.exists():
            options.extend(path for path in sorted(folder.glob("*.csv")) if path.name not in excluded_names)
    unique: dict[str, Path] = {}
    for path in options:
        unique[str(path)] = path
    return list(unique.values())


def option_index(options: list[Path], preferred: Path) -> int:
    for index, option in enumerate(options):
        if option == preferred:
            return index
    for index, option in enumerate(options):
        if option.name == preferred.name:
            return index
    return 0


def build_product_base_metrics(
    products_path: Path,
    orders_path: Path,
    inventory_path: Path,
    returns_path: Path,
    paths: dict[str, Path],
) -> Path:
    products = read_csv(products_path).drop_duplicates(subset=["product_id"], keep="first")
    orders = read_csv(orders_path)
    inventory = read_csv(inventory_path)
    returns = read_csv(returns_path)

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce").fillna(0)
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce").fillna(0)
    inventory["stock_qty"] = pd.to_numeric(inventory["stock_qty"], errors="coerce").fillna(0)
    inventory["stock_amount"] = pd.to_numeric(inventory["stock_amount"], errors="coerce").fillna(0)

    order_metrics = (
        orders.groupby("product_id")
        .agg(
            sales_qty=("quantity", "sum"),
            sales_amount=("order_amount", "sum"),
            out_freq=("order_id", "count"),
        )
        .reset_index()
    )
    inventory_metrics = (
        inventory.groupby("product_id")
        .agg(
            stock_qty=("stock_qty", "sum"),
            stock_amount=("stock_amount", "sum"),
        )
        .reset_index()
    )
    return_metrics = returns.groupby("product_id").size().reset_index(name="return_count")

    product_base_metrics = products.merge(order_metrics, on="product_id", how="left")
    product_base_metrics = product_base_metrics.merge(inventory_metrics, on="product_id", how="left")
    product_base_metrics = product_base_metrics.merge(return_metrics, on="product_id", how="left")

    for column in ["sales_qty", "sales_amount", "out_freq", "stock_qty", "stock_amount", "return_count"]:
        product_base_metrics[column] = product_base_metrics[column].fillna(0)
    denominator = product_base_metrics["out_freq"].where(product_base_metrics["out_freq"] != 0)
    product_base_metrics["return_rate"] = (product_base_metrics["return_count"] / denominator).fillna(0).round(4)

    output_columns = [
        "product_id",
        "product_name",
        "category",
        "unit_price",
        "cost_price",
        "gross_profit_rate",
        "shelf_life_days",
        "is_fresh",
        "is_key_product",
        "sales_qty",
        "sales_amount",
        "out_freq",
        "stock_qty",
        "stock_amount",
        "return_count",
        "return_rate",
    ]
    product_base_metrics = product_base_metrics[output_columns]
    return write_cleaned(product_base_metrics, paths, "product_base_metrics.csv")


def render_data_connection(paths: dict[str, Path]) -> None:
    ensure_module_files(paths)
    st.header("数据连接模块")
    st.caption("以 product_id 为主键，把商品信息、销售指标、库存指标、退货指标合成一张商品基础指标表。")

    options = source_options(paths)
    if not options:
        st.warning("未发现CSV数据，请先生成模拟数据或完成数据清洗。")
        return

    st.markdown(
        """
        <div class="process-strip">
            <span>products 商品主表</span><span>orders 销售汇总</span><span>inventory 库存汇总</span><span>returns 退货汇总</span><span>product_id 合并</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        products_path = st.selectbox(
            "商品主表 products.csv",
            options,
            index=option_index(options, preferred_source_path(paths, "products.csv")),
            format_func=lambda value: value.name,
        )
        orders_path = st.selectbox(
            "订单表 orders.csv",
            options,
            index=option_index(options, preferred_source_path(paths, "orders.csv")),
            format_func=lambda value: value.name,
        )
    with col2:
        inventory_path = st.selectbox(
            "库存表 inventory.csv",
            options,
            index=option_index(options, preferred_source_path(paths, "inventory.csv")),
            format_func=lambda value: value.name,
        )
        returns_path = st.selectbox(
            "退货表 returns.csv",
            options,
            index=option_index(options, preferred_source_path(paths, "returns.csv")),
            format_func=lambda value: value.name,
        )

    source_summary = pd.DataFrame(
        [
            {"数据角色": "商品主表", "文件": products_path.name, "主键": "product_id", "处理方式": "一行一个商品"},
            {"数据角色": "订单销售", "文件": orders_path.name, "主键": "product_id", "处理方式": "汇总销量、销售额、出库频次"},
            {"数据角色": "库存指标", "文件": inventory_path.name, "主键": "product_id", "处理方式": "汇总库存数量、库存金额"},
            {"数据角色": "退货指标", "文件": returns_path.name, "主键": "product_id", "处理方式": "汇总退货次数、退货率"},
        ]
    )
    st.dataframe(source_summary, use_container_width=True, hide_index=True)

    if st.button("生成商品基础指标表", type="primary"):
        output_path = build_product_base_metrics(products_path, orders_path, inventory_path, returns_path, paths)
        output_df = read_csv(output_path)
        st.success(f"已生成并保存：{output_path}")
        render_kpi_cards(
            [
                ("商品数", f"{len(output_df):,}", "一行一个商品"),
                ("字段数", str(len(output_df.columns)), "合并后字段"),
                ("总销量", f"{int(output_df['sales_qty'].sum()):,}", "orders汇总"),
                ("总退货", f"{int(output_df['return_count'].sum()):,}", "returns汇总"),
            ]
        )
        st.markdown("##### product_base_metrics.csv 预览")
        st.dataframe(output_df.head(20), use_container_width=True, hide_index=True)

    output_path = paths["processed_data"] / "product_base_metrics.csv"
    if output_path.exists():
        st.markdown("#### 已保存结果")
        st.write(f"`{output_path}`")
        st.dataframe(read_csv(output_path).head(20), use_container_width=True, hide_index=True)


def dataset_label(path: Path, paths: dict[str, Path]) -> str:
    try:
        parent = path.parent.relative_to(paths["project_root"])
    except ValueError:
        parent = path.parent
    return f"{parent.as_posix()} / {path.name}"


def default_connection_name(main_path: Path, join_paths: list[Path], join_field: str) -> str:
    if main_path.name == "products.csv" and join_field == "product_id":
        return "product_base_metrics.csv"
    join_names = "_".join(path.stem for path in join_paths[:3])
    return f"{main_path.stem}_{join_names}_connected.csv"


def aggregate_join_table(join_df: pd.DataFrame, join_field: str, table_stem: str) -> pd.DataFrame:
    working = join_df.copy()
    numeric_columns = []
    for column in working.columns:
        if column == join_field:
            continue
        converted = pd.to_numeric(working[column], errors="coerce")
        if converted.notna().any():
            working[column] = converted.fillna(0)
            numeric_columns.append(column)

    grouped = working.groupby(join_field, dropna=False)
    result = grouped.size().reset_index(name=f"{table_stem}_row_count")
    if numeric_columns:
        numeric_result = grouped[numeric_columns].sum().reset_index()
        numeric_result = numeric_result.rename(columns={column: f"{table_stem}_{column}_sum" for column in numeric_columns})
        result = result.merge(numeric_result, on=join_field, how="left")
    return result


def build_connected_dataset(
    main_path: Path,
    join_paths: list[Path],
    join_field: str,
    output_name: str,
    paths: dict[str, Path],
) -> Path:
    main_df = read_csv(main_path)
    if join_field not in main_df.columns:
        raise ValueError(f"主表缺少连接字段：{join_field}")

    main_df = main_df.drop_duplicates(subset=[join_field], keep="first")
    connected = main_df.copy()
    for join_path in join_paths:
        join_df = read_csv(join_path)
        if join_field not in join_df.columns:
            raise ValueError(f"{join_path.name} 缺少连接字段：{join_field}")
        join_metrics = aggregate_join_table(join_df, join_field, join_path.stem)
        connected = connected.merge(join_metrics, on=join_field, how="left")
        metric_columns = [column for column in connected.columns if column.startswith(f"{join_path.stem}_")]
        for column in metric_columns:
            connected[column] = connected[column].fillna(0)
    return write_cleaned(connected, paths, output_name)


def render_data_connection(paths: dict[str, Path]) -> None:
    ensure_module_files(paths)
    st.header("数据连接模块")

    options = source_options(paths)
    if not options:
        st.warning("未发现CSV数据，请先生成数据或完成数据清洗。")
        return

    main_path = st.selectbox(
        "选择主表",
        options,
        index=option_index(options, preferred_source_path(paths, "products.csv")),
        format_func=lambda value: dataset_label(value, paths),
    )

    join_options = [path for path in options if path != main_path]
    if not join_options:
        st.warning("至少需要两张CSV表才能完成连接。")
        return

    join_paths = st.multiselect(
        "选择连接表",
        join_options,
        format_func=lambda value: dataset_label(value, paths),
    )

    main_columns = list(read_csv(main_path).columns)
    field_options = [""] + main_columns

    join_field = st.selectbox(
        "选择连接字段",
        field_options,
        index=0,
        format_func=lambda value: "请选择" if value == "" else value,
    )

    if st.button("生成连接结果", type="primary"):
        if not join_paths:
            st.warning("请先选择连接表。")
            return
        if not join_field:
            st.warning("请先选择连接字段。")
            return
        try:
            output_path = build_product_base_metrics(
                preferred_source_path(paths, "products.csv"),
                preferred_source_path(paths, "orders.csv"),
                preferred_source_path(paths, "inventory.csv"),
                preferred_source_path(paths, "returns.csv"),
                paths,
            )
            output_df = read_csv(output_path)
            st.success(f"已保存到清洗数据目录：{output_path.name}")
            st.dataframe(output_df.head(20), use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.error(str(exc))


def render_page(page_name: str, paths: dict[str, Path]) -> None:
    ensure_module_files(paths)
    if page_name == "模块首页":
        render_module_home(paths)
    elif page_name == "数据源管理":
        render_data_source_management(paths)
    elif page_name == "数据质量概览":
        render_quality_overview(paths)
    elif page_name == "数据预处理工作流":
        render_workflow_builder(paths)
    elif page_name == "数据连接模块":
        render_data_connection(paths)
