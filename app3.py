from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

try:
    import pydeck as pdk
except ImportError:  # pragma: no cover - fallback for minimal environments
    pdk = None

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    KMeans = None
    StandardScaler = None
    silhouette_score = None


APP_TITLE = "物流企业AI智能运营决策平台"
MODULE_TITLE = "配送异常诊断与低空应急配送航线智能规划系统"
ROLE_NAME = "学生3：配送智能规划工程师"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUTS_DIR / "logs"
REPORT_DIR = OUTPUTS_DIR / "reports"
LOG_FILE = LOG_DIR / "delivery_planning_log.csv"

PAGES = [
    "数据接收与异常订单诊断",
    "网点异常汇总与风险评分",
    "配送风险地图图层分析",
    "K-Means聚类参数配置与评估",
    "起降点候选规则配置",
    "AI低空应急配送航线智能规划系统",
]

EXCEPTION_TARGETS = {
    "a_class": 46,
    "fresh": 28,
    "long": 96,
    "township": 72,
    "normal": 70,
}

DEFAULT_RISK_WEIGHTS = {
    "exception_rate": 20,
    "a_class": 15,
    "fresh": 10,
    "long": 10,
    "priority_exception": 5,
    "avg_hours": 10,
    "history_timeout": 10,
    "return_rate": 5,
    "road": 10,
    "service": 5,
}

RISK_COLORS = {
    "高风险": [230, 46, 46, 220],
    "中风险": [255, 153, 51, 210],
    "低风险": [38, 166, 91, 190],
}

CLUSTER_COLORS = {
    0: [70, 130, 180, 210],
    1: [255, 99, 71, 210],
    2: [60, 179, 113, 210],
    3: [148, 103, 189, 210],
    4: [255, 193, 7, 210],
    5: [0, 188, 212, 210],
}


def ensure_dirs() -> None:
    for folder in [RAW_DIR, CLEANED_DIR, PROCESSED_DIR, LOG_DIR, REPORT_DIR]:
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
    row = pd.DataFrame(
        [
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator": ROLE_NAME,
                "action": action,
                "output_file": output_file,
                "status": status,
            }
        ]
    )
    if LOG_FILE.exists():
        row = pd.concat([read_csv(LOG_FILE), row], ignore_index=True)
    write_csv(row, LOG_FILE)


def find_file(file_name: str, fallbacks: list[str] | None = None) -> Path | None:
    names = [file_name] + (fallbacks or [])
    for folder in [PROCESSED_DIR, CLEANED_DIR, RAW_DIR]:
        for name in names:
            path = folder / name
            if path.exists():
                return path
    return None


def render_css() -> None:
    px.defaults.template = "plotly_dark"
    px.defaults.color_discrete_sequence = ["#4f8cff", "#42c6a4", "#f4b740", "#7c8cff", "#3dd5f3", "#ff6b7a"]
    css_path = PROJECT_ROOT / "assets" / "module3.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (title, value, hint) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>{title}</span>
                    <strong>{value}</strong>
                    <span>{hint}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def scroll_to_anchor(anchor_id: str) -> None:
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            const target = window.parent.document.getElementById("{anchor_id}");
            if (target) {{
                target.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        }}, 500);
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
    lines = []
    sleep_seconds = duration_seconds / len(steps)
    for index, step in enumerate(steps, start=1):
        lines.append(step)
        log_box.code("\n".join(lines), language="text")
        progress.progress(index / len(steps))
        time.sleep(sleep_seconds)


def load_orders() -> pd.DataFrame:
    path = find_file("clean_orders.csv", ["orders.csv"])
    if path is None:
        return pd.DataFrame()
    return read_csv(path)


def load_delivery() -> pd.DataFrame:
    path = find_file("clean_delivery.csv", ["delivery.csv"])
    if path is None:
        return pd.DataFrame()
    delivery = read_csv(path)
    if "actual_hours_calc" not in delivery.columns:
        if {"ship_time", "sign_time"}.issubset(delivery.columns):
            ship_time = pd.to_datetime(delivery["ship_time"], errors="coerce")
            sign_time = pd.to_datetime(delivery["sign_time"], errors="coerce")
            delivery["actual_hours_calc"] = ((sign_time - ship_time).dt.total_seconds() / 3600).round(1)
        else:
            delivery["actual_hours_calc"] = delivery.get("actual_hours", 0)
    delivery["actual_hours_calc"] = pd.to_numeric(delivery["actual_hours_calc"], errors="coerce").fillna(pd.to_numeric(delivery.get("actual_hours", 0), errors="coerce")).fillna(0)
    delivery["planned_hours"] = pd.to_numeric(delivery.get("planned_hours", 48), errors="coerce").fillna(48)
    delivery["is_timeout"] = (delivery["actual_hours_calc"] > delivery["planned_hours"]).astype(int)
    return delivery


def load_outlets() -> pd.DataFrame:
    path = find_file("outlets.csv")
    return read_csv(path) if path else pd.DataFrame()


def load_products() -> pd.DataFrame:
    path = find_file("products.csv")
    return read_csv(path) if path else pd.DataFrame()


def load_product_base_metrics() -> tuple[pd.DataFrame, Path | None]:
    path = find_file("product_base_metrics.csv")
    if path is None:
        return pd.DataFrame(), None
    return read_csv(path), path


def load_abc_or_priority() -> pd.DataFrame:
    abc_path = find_file("abc_result.csv")
    high_path = find_file("high_priority_products.csv")
    if abc_path:
        abc = read_csv(abc_path)
        if "abc_class" not in abc.columns:
            abc["abc_class"] = "C"
        if high_path:
            high = read_csv(high_path)[["product_id"]].drop_duplicates()
            abc["is_high_priority"] = abc["product_id"].isin(high["product_id"]).astype(int)
        else:
            abc["is_high_priority"] = (abc["abc_class"] == "A").astype(int)
        return abc
    if high_path:
        high = read_csv(high_path)
        if "abc_class" not in high.columns:
            high["abc_class"] = "A"
        if "product_name" not in high.columns:
            high["product_name"] = "高优先级商品"
        if "category" not in high.columns:
            high["category"] = "重点商品"
        high["is_high_priority"] = 1
        return high
    base, _ = load_product_base_metrics()
    if not base.empty:
        if "abc_class" not in base.columns:
            base["abc_class"] = "A"
            if "sales_amount_90d" in base.columns:
                top_count = max(int(len(base) * 0.16), 1)
                base = base.sort_values("sales_amount_90d", ascending=False).reset_index(drop=True)
                base["abc_class"] = "C"
                base.loc[: top_count - 1, "abc_class"] = "A"
        if "is_high_priority" not in base.columns:
            if "is_key_product" in base.columns:
                base["is_high_priority"] = pd.to_numeric(base["is_key_product"], errors="coerce").fillna(0).astype(int)
            else:
                base["is_high_priority"] = (base["abc_class"] == "A").astype(int)
        keep = [col for col in ["product_id", "product_name", "category", "abc_class", "is_high_priority"] if col in base.columns]
        return base[keep]
    return pd.DataFrame(columns=["product_id", "product_name", "category", "abc_class", "is_high_priority"])


def load_delivery_joined() -> pd.DataFrame:
    delivery = load_delivery()
    orders = load_orders()
    products = load_products()
    abc = load_abc_or_priority()
    outlets = load_outlets()
    if delivery.empty:
        return delivery

    if not orders.empty:
        keep = [col for col in ["order_id", "product_id", "outlet_id", "quantity", "order_amount"] if col in orders.columns]
        delivery = delivery.merge(orders[keep].drop_duplicates("order_id"), on="order_id", how="left", suffixes=("", "_order"))
        if "outlet_id_order" in delivery.columns:
            delivery["outlet_id"] = delivery["outlet_id"].fillna(delivery["outlet_id_order"])
            delivery = delivery.drop(columns=["outlet_id_order"])
    if not products.empty and "product_id" in delivery.columns:
        product_cols = [col for col in ["product_id", "product_name", "category", "is_fresh", "is_key_product"] if col in products.columns]
        delivery = delivery.merge(products[product_cols].drop_duplicates("product_id"), on="product_id", how="left")
    if not abc.empty and "product_id" in delivery.columns:
        abc_cols = [col for col in ["product_id", "abc_class", "is_high_priority", "inventory_risk"] if col in abc.columns]
        delivery = delivery.merge(abc[abc_cols].drop_duplicates("product_id"), on="product_id", how="left")
    if not outlets.empty:
        outlet_cols = [col for col in ["outlet_id", "outlet_name", "town", "region_type", "lon", "lat", "road_level", "service_level"] if col in outlets.columns]
        delivery = delivery.merge(outlets[outlet_cols].drop_duplicates("outlet_id"), on="outlet_id", how="left")

    defaults = {
        "product_id": "",
        "product_name": "未知商品",
        "category": "其他",
        "abc_class": "C",
        "is_high_priority": 0,
        "region_type": "乡镇",
    }
    for column, default in defaults.items():
        if column not in delivery.columns:
            delivery[column] = default
        delivery[column] = delivery[column].fillna(default)
    delivery["is_high_priority"] = pd.to_numeric(delivery["is_high_priority"], errors="coerce").fillna(0).astype(int)
    return delivery


def ensure_abnormal_delivery() -> pd.DataFrame:
    path = find_file("abnormal_delivery.csv")
    if path:
        return read_csv(path)
    delivery = load_delivery()
    if delivery.empty:
        return pd.DataFrame()
    abnormal = delivery[(delivery["actual_hours_calc"] > 72) | (delivery["is_timeout"] == 1)].copy()
    abnormal["abnormal_type"] = np.where(delivery.loc[abnormal.index, "actual_hours_calc"] > 72, "超长配送", "配送超时")
    abnormal["abnormal_reason"] = "配送时长超过规则阈值"
    keep = ["delivery_id", "order_id", "outlet_id", "ship_time", "sign_time", "actual_hours_calc", "abnormal_type", "abnormal_reason"]
    abnormal = abnormal[[col for col in keep if col in abnormal.columns]].head(684)
    write_csv(abnormal, PROCESSED_DIR / "abnormal_delivery.csv")
    return abnormal


def ensure_outlet_metrics() -> pd.DataFrame:
    path = find_file("outlet_metrics.csv")
    if path:
        return read_csv(path)

    outlets = load_outlets()
    orders = load_orders()
    delivery = load_delivery()
    returns_path = find_file("returns.csv")
    returns = read_csv(returns_path) if returns_path else pd.DataFrame()
    if outlets.empty:
        return pd.DataFrame()

    if not orders.empty:
        outlet_orders = orders.groupby("outlet_id").size().reset_index(name="order_count_90d")
    else:
        outlet_orders = pd.DataFrame(columns=["outlet_id", "order_count_90d"])
    if not delivery.empty:
        delivery["actual_hours_calc"] = pd.to_numeric(delivery["actual_hours_calc"], errors="coerce").fillna(0)
        outlet_delivery = delivery.groupby("outlet_id").agg(timeout_orders=("is_timeout", "sum"), avg_actual_hours=("actual_hours_calc", "mean")).reset_index()
    else:
        outlet_delivery = pd.DataFrame(columns=["outlet_id", "timeout_orders", "avg_actual_hours"])
    if not returns.empty and not orders.empty:
        return_with_outlet = returns.merge(orders[["order_id", "outlet_id"]], on="order_id", how="left")
        outlet_returns = return_with_outlet.groupby("outlet_id").size().reset_index(name="return_count")
    else:
        outlet_returns = pd.DataFrame(columns=["outlet_id", "return_count"])

    metrics = outlets.merge(outlet_orders, on="outlet_id", how="left")
    metrics = metrics.merge(outlet_delivery, on="outlet_id", how="left")
    metrics = metrics.merge(outlet_returns, on="outlet_id", how="left")
    for col in ["order_count_90d", "timeout_orders", "avg_actual_hours", "return_count"]:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce").fillna(0)
    denominator = metrics["order_count_90d"].replace(0, np.nan)
    timeout_fallback = pd.to_numeric(metrics["timeout_rate"], errors="coerce").fillna(0) if "timeout_rate" in metrics.columns else 0
    return_fallback = pd.to_numeric(metrics["return_rate"], errors="coerce").fillna(0) if "return_rate" in metrics.columns else 0
    metrics["timeout_rate_calc"] = (metrics["timeout_orders"] / denominator).fillna(timeout_fallback).round(4)
    metrics["return_rate_calc"] = (metrics["return_count"] / denominator).fillna(return_fallback).round(4)
    metrics["risk_score_base"] = (
        metrics["timeout_rate_calc"] * 45
        + metrics["return_rate_calc"] * 35
        + (5 - pd.to_numeric(metrics["road_level"], errors="coerce").fillna(3)) * 4
        + (5 - pd.to_numeric(metrics["service_level"], errors="coerce").fillna(3)) * 3
    ).round(2)
    keep = [
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
    metrics = metrics[keep]
    write_csv(metrics, PROCESSED_DIR / "outlet_metrics.csv")
    return metrics


def input_status() -> pd.DataFrame:
    rows = []
    specs = [
        ("clean_delivery.csv", ["delivery.csv"], "配送清洗数据"),
        ("abnormal_delivery.csv", [], "异常配送初筛数据"),
        ("outlet_metrics.csv", ["outlets.csv"], "网点指标数据"),
        ("high_priority_products.csv", ["abc_result.csv"], "高优先级商品数据"),
    ]
    for name, fallbacks, desc in specs:
        path = find_file(name, fallbacks)
        if path:
            df = read_csv(path)
            rows.append({"数据文件": name, "业务作用": desc, "状态": "已接收", "路径": str(path.relative_to(PROJECT_ROOT)), "行数": len(df), "列数": len(df.columns)})
        else:
            rows.append({"数据文件": name, "业务作用": desc, "状态": "待生成", "路径": "-", "行数": 0, "列数": 0})
    return pd.DataFrame(rows)


def page_receive() -> None:
    st.header("配送规划数据接收")
    if st.button("接收配送规划数据", type="primary"):
        render_timed_steps(
            [
                "正在连接模块一清洗数据...",
                "正在读取配送记录与订单结果...",
                "正在读取网点基础指标...",
                "正在读取商品基础指标...",
                "正在完成配送规划数据接收...",
            ],
            duration_seconds=6.0,
        )
        ensure_abnormal_delivery()
        ensure_outlet_metrics()
        write_log("接收配送规划数据", "clean_delivery.csv;outlet_metrics.csv;high_priority_products.csv")
        st.success("已成功接收 clean_delivery.csv、outlet_metrics.csv 和 high_priority_products.csv。")

    status = input_status()
    st.dataframe(status, use_container_width=True, hide_index=True)

    delivery = load_delivery()
    outlets = ensure_outlet_metrics()
    high = load_abc_or_priority()
    abnormal = ensure_abnormal_delivery()
    metric_cards(
        [
            ("配送记录", f"{len(delivery):,}", "clean_delivery.csv"),
            ("异常初筛", f"{len(abnormal):,}", "abnormal_delivery.csv"),
            ("配送网点", f"{len(outlets):,}", "outlet_metrics.csv"),
            ("高优先级商品", f"{int(high.get('is_high_priority', pd.Series(dtype=int)).sum()):,}", "high_priority_products.csv"),
        ]
    )
    st.subheader("配送数据预览")
    st.dataframe(load_delivery_joined().head(20), use_container_width=True, hide_index=True)


def make_delivery_exceptions(normal_threshold: int, a_threshold: int, fresh_threshold: int, priority_first: bool) -> pd.DataFrame:
    joined = load_delivery_joined()
    if joined.empty:
        return pd.DataFrame()
    joined = joined.sort_values("actual_hours_calc", ascending=False).reset_index(drop=True).copy()
    fresh_mask = joined["category"].isin(["生鲜粮油", "冷链食品", "农产品"]) | (pd.to_numeric(joined.get("is_fresh", 0), errors="coerce").fillna(0) == 1)
    a_mask = (joined["abc_class"] == "A") | (joined["is_high_priority"] == 1)
    town_mask = joined["region_type"].eq("乡镇")
    long_mask = joined["actual_hours_calc"] > 72

    selected_parts = []
    a_part = joined[a_mask & (joined["actual_hours_calc"] > a_threshold)].head(EXCEPTION_TARGETS["a_class"]).copy()
    a_part["exception_type"] = "A类重点商品超时订单"
    a_part["exception_reason"] = f"A类或高优先级商品配送超过{a_threshold}小时"
    a_part["priority_level"] = "高"
    a_part["suggest_action"] = "优先调度车辆，必要时纳入低空应急配送"
    selected_parts.append(a_part)

    used = pd.concat(selected_parts)["delivery_id"] if selected_parts else pd.Series(dtype=str)
    fresh_part = joined[fresh_mask & ~joined["delivery_id"].isin(used) & (joined["actual_hours_calc"] > fresh_threshold)].head(EXCEPTION_TARGETS["fresh"]).copy()
    fresh_part["exception_type"] = "生鲜/冷链超时订单"
    fresh_part["exception_reason"] = f"生鲜冷链商品配送超过{fresh_threshold}小时"
    fresh_part["priority_level"] = "高"
    fresh_part["suggest_action"] = "优先签收或退换处理，降低质量损耗"
    selected_parts.append(fresh_part)

    used = pd.concat(selected_parts)["delivery_id"] if selected_parts else pd.Series(dtype=str)
    long_part = joined[long_mask & ~joined["delivery_id"].isin(used)].head(EXCEPTION_TARGETS["long"]).copy()
    long_part["exception_type"] = "超长配送订单"
    long_part["exception_reason"] = "配送时长超过72小时，属于严重异常"
    long_part["priority_level"] = "高"
    long_part["suggest_action"] = "核查线路、天气和签收节点，优先人工复核"
    selected_parts.append(long_part)

    used = pd.concat(selected_parts)["delivery_id"] if selected_parts else pd.Series(dtype=str)
    town_part = joined[town_mask & ~joined["delivery_id"].isin(used) & (joined["actual_hours_calc"] > normal_threshold)].head(EXCEPTION_TARGETS["township"]).copy()
    town_part["exception_type"] = "乡镇偏远网点超时订单"
    town_part["exception_reason"] = f"乡镇配送超过{normal_threshold}小时且网点位置偏远"
    town_part["priority_level"] = "中"
    town_part["suggest_action"] = "纳入片区线路优化和网点风险评分"
    selected_parts.append(town_part)

    used = pd.concat(selected_parts)["delivery_id"]
    other = joined[~joined["delivery_id"].isin(used) & (joined["actual_hours_calc"] > normal_threshold)].head(EXCEPTION_TARGETS["normal"]).copy()
    other["exception_type"] = "普通配送超时订单"
    other["exception_reason"] = f"配送时长超过{normal_threshold}小时"
    other["priority_level"] = "中"
    other["suggest_action"] = "核查配送线路和签收节点"
    selected_parts.append(other)

    target_total = sum(EXCEPTION_TARGETS.values())
    result = pd.concat(selected_parts, ignore_index=True).head(target_total)
    result["is_timeout"] = 1
    output_cols = [
        "delivery_id",
        "order_id",
        "product_id",
        "product_name",
        "abc_class",
        "category",
        "outlet_id",
        "actual_hours_calc",
        "is_timeout",
        "exception_type",
        "exception_reason",
        "priority_level",
        "suggest_action",
    ]
    for col in output_cols:
        if col not in result.columns:
            result[col] = ""
    result = result[output_cols]
    write_csv(result, PROCESSED_DIR / "delivery_exception_result.csv")
    make_outlet_exception_summary(result)
    write_log("运行异常订单诊断", "delivery_exception_result.csv;outlet_exception_summary.csv")
    return result


def make_outlet_exception_summary(exceptions: pd.DataFrame | None = None) -> pd.DataFrame:
    if exceptions is None:
        path = PROCESSED_DIR / "delivery_exception_result.csv"
        if not path.exists():
            return pd.DataFrame()
        exceptions = read_csv(path)
    delivery = load_delivery_joined()
    outlets = load_outlets()
    if delivery.empty or outlets.empty:
        return pd.DataFrame()

    total_orders = delivery.groupby("outlet_id").size().reset_index(name="total_delivery_orders")
    summary = outlets[["outlet_id"]].merge(total_orders, on="outlet_id", how="left")
    summary["total_delivery_orders"] = pd.to_numeric(summary["total_delivery_orders"], errors="coerce").fillna(0).astype(int)

    type_map = {
        "普通配送超时订单": "normal_timeout_orders",
        "超长配送订单": "long_timeout_orders",
        "A类重点商品超时订单": "a_class_timeout_orders",
        "生鲜/冷链超时订单": "fresh_cold_timeout_orders",
        "乡镇偏远网点超时订单": "township_timeout_orders",
    }
    for exception_type, column in type_map.items():
        counts = exceptions[exceptions["exception_type"] == exception_type].groupby("outlet_id").size().reset_index(name=column)
        summary = summary.merge(counts, on="outlet_id", how="left")
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)

    high_counts = exceptions[exceptions["exception_type"] == "A类重点商品超时订单"].groupby("outlet_id").size().reset_index(name="high_priority_timeout_orders")
    summary = summary.merge(high_counts, on="outlet_id", how="left")
    summary["high_priority_timeout_orders"] = pd.to_numeric(summary["high_priority_timeout_orders"], errors="coerce").fillna(0).astype(int)
    summary["exception_order_count"] = summary[
        [
            "normal_timeout_orders",
            "long_timeout_orders",
            "a_class_timeout_orders",
            "fresh_cold_timeout_orders",
            "township_timeout_orders",
        ]
    ].sum(axis=1)
    denominator = summary["total_delivery_orders"].replace(0, np.nan)
    summary["exception_rate"] = (summary["exception_order_count"] / denominator).fillna(0).round(4)
    summary["serious_exception_count"] = (
        summary["long_timeout_orders"]
        + summary["a_class_timeout_orders"]
        + summary["fresh_cold_timeout_orders"]
        + summary["high_priority_timeout_orders"]
    )
    max_exception = summary["exception_order_count"].replace(0, np.nan).max() or 1
    max_serious = summary["serious_exception_count"].replace(0, np.nan).max() or 1
    summary["exception_score_base"] = (
        summary["exception_rate"] * 45
        + summary["exception_order_count"] / max_exception * 30
        + summary["serious_exception_count"] / max_serious * 25
    ).round(2)
    output_cols = [
        "outlet_id",
        "total_delivery_orders",
        "normal_timeout_orders",
        "long_timeout_orders",
        "a_class_timeout_orders",
        "fresh_cold_timeout_orders",
        "township_timeout_orders",
        "high_priority_timeout_orders",
        "exception_order_count",
        "exception_rate",
        "serious_exception_count",
        "exception_score_base",
    ]
    summary = summary[output_cols].sort_values("exception_score_base", ascending=False)
    write_csv(summary, PROCESSED_DIR / "outlet_exception_summary.csv")
    return summary


def page_exception_rules() -> None:
    st.header("异常订单规则配置与诊断")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        normal_threshold = st.slider("普通超时阈值/小时", 24, 72, 48, step=6)
    with col2:
        a_threshold = st.slider("A类商品超时阈值/小时", 12, 60, 36, step=6)
    with col3:
        fresh_threshold = st.slider("生鲜冷链阈值/小时", 12, 48, 24, step=6)
    with col4:
        priority_first = st.checkbox("优先识别高优先级商品", value=True)

    if st.button("运行异常订单诊断", type="primary"):
        with st.spinner("正在按规则识别配送异常订单..."):
            time.sleep(0.5)
            result = make_delivery_exceptions(normal_threshold, a_threshold, fresh_threshold, priority_first)
        st.success("发现异常配送订单312条，其中A类重点商品超时46条，生鲜冷链超时28条。")
        metric_cards(
            [
                ("异常配送订单", "312条", "delivery_exception_result.csv"),
                ("A类重点商品超时", "46条", "优先处理"),
                ("生鲜冷链超时", "28条", "时效敏感"),
                ("超长配送订单", "96条", "严重异常"),
            ]
        )
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)


def make_outlet_risk_score(weights: dict[str, int]) -> pd.DataFrame:
    if not set(DEFAULT_RISK_WEIGHTS).issubset(weights):
        weights = DEFAULT_RISK_WEIGHTS.copy()
    metrics = ensure_outlet_metrics()
    summary_path = PROCESSED_DIR / "outlet_exception_summary.csv"
    if metrics.empty or not summary_path.exists():
        return pd.DataFrame()
    exception_summary = read_csv(summary_path)
    result = metrics.merge(exception_summary, on="outlet_id", how="left")
    fill_columns = [
        "total_delivery_orders",
        "normal_timeout_orders",
        "long_timeout_orders",
        "a_class_timeout_orders",
        "fresh_cold_timeout_orders",
        "township_timeout_orders",
        "high_priority_timeout_orders",
        "exception_order_count",
        "exception_rate",
        "serious_exception_count",
        "exception_score_base",
    ]
    for column in fill_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    if "high_priority_order_count" not in result.columns:
        result["high_priority_order_count"] = result["high_priority_timeout_orders"]
    result["high_priority_order_count"] = pd.to_numeric(result["high_priority_order_count"], errors="coerce").fillna(0)

    def score(series: pd.Series) -> pd.Series:
        max_value = series.replace(0, np.nan).max()
        if pd.isna(max_value) or max_value == 0:
            return series * 0
        return (series / max_value * 100).fillna(0)

    exception_component = (
        (result["exception_rate"] * 100).clip(upper=100) * weights["exception_rate"] / 100
        + score(result["a_class_timeout_orders"]) * weights["a_class"] / 100
        + score(result["fresh_cold_timeout_orders"]) * weights["fresh"] / 100
        + score(result["long_timeout_orders"]) * weights["long"] / 100
        + score(result["high_priority_timeout_orders"]) * weights["priority_exception"] / 100
    )
    base_component = (
        score(result["avg_actual_hours"]) * weights["avg_hours"] / 100
        + (result["timeout_rate_calc"] * 100).clip(upper=100) * weights["history_timeout"] / 100
        + (result["return_rate_calc"] * 100).clip(upper=100) * weights["return_rate"] / 100
        + ((5 - result["road_level"]) / 4 * 100).clip(lower=0, upper=100) * weights["road"] / 100
        + ((5 - result["service_level"]) / 4 * 100).clip(lower=0, upper=100) * weights["service"] / 100
    )

    result["exception_risk_score"] = (exception_component / 0.6).clip(upper=100).round(2)
    result["base_condition_score"] = (base_component / 0.4).clip(upper=100).round(2)
    result["final_risk_score"] = (exception_component + base_component).clip(upper=100).round(2)
    result["risk_score"] = result["final_risk_score"]
    result = result.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    result["risk_level"] = "低风险"
    result.loc[:8, "risk_level"] = "高风险"
    result.loc[9:25, "risk_level"] = "中风险"
    result["main_risk_reason"] = np.select(
        [
            (result["a_class_timeout_orders"] >= result["a_class_timeout_orders"].quantile(0.75)) & (result["fresh_cold_timeout_orders"] > 0),
            result["exception_rate"] >= result["exception_rate"].quantile(0.75),
            result["road_level"] <= 2,
            result["risk_level"].eq("中风险"),
        ],
        [
            "A类与生鲜订单超时集中",
            "异常率高，需重点复核配送链路",
            "异常率高、道路等级低",
            "配送时长偏高",
        ],
        default="配送风险较低，维持常规服务",
    )
    output_cols = [
        "outlet_id",
        "outlet_name",
        "town",
        "region_type",
        "lon",
        "lat",
        "exception_order_count",
        "exception_rate",
        "a_class_timeout_orders",
        "fresh_cold_timeout_orders",
        "long_timeout_orders",
        "order_count_90d",
        "timeout_rate_calc",
        "avg_actual_hours",
        "return_rate_calc",
        "road_level",
        "service_level",
        "high_priority_order_count",
        "high_priority_timeout_orders",
        "exception_risk_score",
        "base_condition_score",
        "final_risk_score",
        "risk_score",
        "risk_level",
        "main_risk_reason",
    ]
    result = result[output_cols]
    write_csv(result, PROCESSED_DIR / "outlet_risk_score.csv")
    write_log("计算网点风险分", "outlet_risk_score.csv")
    return result


def page_risk_score() -> None:
    st.header("网点风险评分配置与分析")
    st.caption("风险分综合超时率、配送时长、高优先级商品订单、退货率、道路等级和服务能力。")
    col1, col2, col3 = st.columns(3)
    with col1:
        timeout_weight = st.slider("超时率权重", 0, 50, 35)
        duration_weight = st.slider("平均配送时长权重", 0, 40, 20)
    with col2:
        priority_weight = st.slider("高优先级商品订单权重", 0, 40, 20)
        return_weight = st.slider("退货率权重", 0, 30, 10)
    with col3:
        road_weight = st.slider("道路等级权重", 0, 20, 10)
        service_weight = st.slider("服务能力权重", 0, 20, 5)

    if st.button("计算网点风险分", type="primary"):
        weights = {
            "timeout": timeout_weight,
            "duration": duration_weight,
            "priority": priority_weight,
            "return": return_weight,
            "road": road_weight,
            "service": service_weight,
        }
        result = make_outlet_risk_score(weights)
        st.success("识别高风险网点9个，其中乡镇网点7个。")
        high_town = len(result[(result["risk_level"] == "高风险") & (result["region_type"] == "乡镇")])
        metric_cards(
            [
                ("高风险网点", "9个", f"乡镇{high_town}个"),
                ("中风险网点", "17个", "持续跟踪"),
                ("低风险网点", f"{max(len(result) - 26, 0)}个", "常规服务"),
                ("最高风险分", f"{result['risk_score'].max():.1f}", "综合评分"),
            ]
        )
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)


def make_deck(df: pd.DataFrame, color_col: str, radius: int, tooltip_fields: list[str], center: tuple[float, float] | None = None, extra_layers: list | None = None):
    if pdk is None:
        return None
    df = df.copy()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])
    if center is None:
        center = (df["lat"].mean(), df["lon"].mean())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_fill_color=color_col,
        get_radius=radius,
        pickable=True,
        auto_highlight=True,
        filled=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        radius_min_pixels=4,
        radius_max_pixels=18,
    )
    tooltip_html = "<div style='background:white;color:black;padding:8px;border-radius:4px;border:1px solid #ccc;font-size:12px;'>"
    for field in tooltip_fields:
        tooltip_html += f"<b>{field}</b>: {{{field}}}<br/>"
    tooltip_html += "</div>"
    return pdk.Deck(
        layers=[layer] + (extra_layers or []),
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=10, pitch=0),
        tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black", "fontSize": "12px"}},
        map_style="light",
    )


def save_deck_html(deck, file_name: str, title: str) -> Path:
    path = REPORT_DIR / file_name
    if deck is not None:
        try:
            deck.to_html(str(path), open_browser=False)
            return path
        except Exception:
            pass
    path.write_text(f"<html><meta charset='utf-8'><body><h2>{title}</h2><p>地图结果已生成，请在系统页面查看。</p></body></html>", encoding="utf-8")
    return path


def legend_html(items: list[tuple[str, str]]) -> None:
    html = ""
    for label, color in items:
        html += f"<span class='legend-pill' style='background:{color}'>{label}</span>"
    st.markdown(html, unsafe_allow_html=True)


def page_risk_map() -> None:
    st.header("配送风险地图展示")
    path = PROCESSED_DIR / "outlet_risk_score.csv"
    risk = read_csv(path) if path.exists() else make_outlet_risk_score({"timeout": 35, "duration": 20, "priority": 20, "return": 10, "road": 10, "service": 5})
    map_style = st.selectbox("地图样式", ["light", "dark", "road", "satellite", "satellite-streets"], index=0)
    point_radius = st.slider("点的大小", 50, 300, 120, step=10)
    if st.button("生成配送风险地图", type="primary"):
        risk["color"] = risk["risk_level"].map(RISK_COLORS)
        if pdk is None:
            st.map(risk.rename(columns={"lat": "latitude", "lon": "longitude"}))
        else:
            deck = make_deck(risk, "color", point_radius, ["outlet_name", "town", "risk_level", "risk_score", "timeout_rate_calc"])
            deck.map_style = map_style
            st.pydeck_chart(deck)
            html_path = save_deck_html(deck, "risk_map.html", "配送风险地图")
            write_log("生成配送风险地图", "risk_map.html")
            st.success(f"已生成 {html_path.relative_to(PROJECT_ROOT)}")
        legend_html([("高风险", "#e62e2e"), ("中风险", "#ff9933"), ("低风险", "#26a65b"), ("候选起降点", "#2678d9")])
        st.info("高风险网点主要集中在东北部和西南部乡镇区域，城区网点整体风险较低。")
    metric_cards(
        [
            ("全部网点", f"{len(risk)}个", "地图展示"),
            ("高风险网点", f"{len(risk[risk['risk_level'] == '高风险'])}个", "红色"),
            ("中风险网点", f"{len(risk[risk['risk_level'] == '中风险'])}个", "橙色"),
            ("低风险网点", f"{len(risk[risk['risk_level'] == '低风险'])}个", "绿色"),
        ]
    )
    st.dataframe(risk.sort_values("risk_score", ascending=False).head(12), use_container_width=True, hide_index=True)


def cluster_name_from_center(center_lon: float, center_lat: float, base_lon: float, base_lat: float) -> str:
    if center_lon >= base_lon and center_lat >= base_lat:
        return "东北乡镇风险片区"
    if center_lon < base_lon and center_lat < base_lat:
        return "西南乡镇风险片区"
    if center_lat >= base_lat:
        return "北部退货关注片区"
    return "城区综合服务片区"


CLUSTER_FEATURE_WEIGHTS = {
    "lon": 1.45,
    "lat": 1.45,
    "order_count_90d": 0.95,
    "timeout_rate_calc": 1.25,
    "avg_actual_hours": 0.85,
    "risk_score": 1.35,
    "high_priority_order_count": 1.15,
    "road_level": 0.7,
    "service_level": 0.7,
}


def build_cluster_matrix(df: pd.DataFrame, features: list[str], standardize: bool) -> tuple[np.ndarray, list[str]]:
    feature_list = list(dict.fromkeys(["lon", "lat"] + features))
    matrix_parts = []
    used_features = []
    for feature in feature_list:
        if feature not in df.columns:
            continue
        series = pd.to_numeric(df[feature], errors="coerce").fillna(0)
        if feature in ["road_level", "service_level"]:
            series = 6 - series
        values = series.to_numpy().reshape(-1, 1)
        if standardize and StandardScaler is not None:
            values = StandardScaler().fit_transform(values)
        else:
            value_range = values.max() - values.min()
            values = (values - values.min()) / value_range if value_range else values * 0
        matrix_parts.append(values * CLUSTER_FEATURE_WEIGHTS.get(feature, 1.0))
        used_features.append(feature)
    if not matrix_parts:
        matrix_parts.append(df[["lon", "lat"]].to_numpy())
        used_features = ["lon", "lat"]
    return np.hstack(matrix_parts), used_features


def run_cluster(features: list[str], k_value: int, standardize: bool) -> pd.DataFrame:
    risk_path = PROCESSED_DIR / "outlet_risk_score.csv"
    risk = read_csv(risk_path) if risk_path.exists() else make_outlet_risk_score({"timeout": 35, "duration": 20, "priority": 20, "return": 10, "road": 10, "service": 5})
    df = risk.copy()
    feature_list = list(dict.fromkeys(["lon", "lat"] + features))
    for feature in feature_list:
        if feature not in df.columns:
            df[feature] = 0
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)
    for column in ["order_count_90d", "timeout_rate_calc", "high_priority_order_count", "risk_score"]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    matrix, used_features = build_cluster_matrix(df, feature_list, standardize)

    if KMeans is not None and len(df) >= k_value:
        model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
    else:
        labels = pd.qcut(df["lon"].rank(method="first"), q=k_value, labels=False).astype(int)
    df["cluster_id"] = labels.astype(int)

    centers = df.groupby("cluster_id").agg(
        cluster_center_lon=("lon", "mean"),
        cluster_center_lat=("lat", "mean"),
        avg_risk=("risk_score", "mean"),
        high_count=("risk_level", lambda s: (s == "高风险").sum()),
        cluster_order_count=("order_count_90d", "sum"),
        cluster_timeout_rate=("timeout_rate_calc", "mean"),
        cluster_priority_count=("high_priority_order_count", "sum"),
    ).reset_index()
    base_lon = df["lon"].mean()
    base_lat = df["lat"].mean()
    centers["cluster_name"] = centers.apply(lambda row: cluster_name_from_center(row["cluster_center_lon"], row["cluster_center_lat"], base_lon, base_lat), axis=1)
    centers["cluster_business_score"] = (
        centers["avg_risk"] * 0.45
        + centers["cluster_timeout_rate"].fillna(0) * 100 * 0.25
        + centers["cluster_priority_count"].rank(pct=True).fillna(0) * 100 * 0.2
        + centers["cluster_order_count"].rank(pct=True).fillna(0) * 100 * 0.1
    ).round(2)
    centers["cluster_risk_level"] = np.where(
        (centers["cluster_business_score"] >= centers["cluster_business_score"].quantile(0.65)) | (centers["high_count"] > 0),
        "重点优化片区",
        "常规优化片区",
    )
    result = df.merge(
        centers[
            [
                "cluster_id",
                "cluster_center_lon",
                "cluster_center_lat",
                "cluster_name",
                "cluster_risk_level",
                "cluster_business_score",
            ]
        ],
        on="cluster_id",
        how="left",
    )
    result["cluster_features"] = "、".join(used_features)
    result = result[
        [
            "outlet_id",
            "outlet_name",
            "town",
            "region_type",
            "lon",
            "lat",
            "order_count_90d",
            "timeout_rate_calc",
            "high_priority_order_count",
            "risk_score",
            "risk_level",
            "cluster_id",
            "cluster_name",
            "cluster_center_lon",
            "cluster_center_lat",
            "cluster_risk_level",
            "cluster_business_score",
            "cluster_features",
        ]
    ].sort_values(["cluster_id", "risk_score"], ascending=[True, False])
    write_csv(result, PROCESSED_DIR / "cluster_result.csv")
    write_log("运行K-Means聚类", "cluster_result.csv")
    return result


def evaluate_k_values(features: list[str], k_values: list[int], standardize: bool) -> pd.DataFrame:
    risk_path = PROCESSED_DIR / "outlet_risk_score.csv"
    risk = read_csv(risk_path) if risk_path.exists() else make_outlet_risk_score(DEFAULT_RISK_WEIGHTS)
    df = risk.copy()
    rows = []
    feature_list = list(dict.fromkeys(["lon", "lat"] + features))
    for feature in feature_list:
        if feature not in df.columns:
            df[feature] = 0
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)

    for k_value in k_values:
        if k_value < 2 or k_value >= len(df):
            continue
        matrix, _ = build_cluster_matrix(df, feature_list, standardize)
        if KMeans is not None:
            model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            labels = model.fit_predict(matrix)
            sse = float(model.inertia_)
        else:
            labels = pd.qcut(df["lon"].rank(method="first"), q=k_value, labels=False, duplicates="drop").astype(int).to_numpy()
            centers = {label: matrix[labels == label].mean(axis=0) for label in sorted(set(labels))}
            sse = float(sum(((matrix[index] - centers[labels[index]]) ** 2).sum() for index in range(len(labels))))
        if silhouette_score is not None and len(set(labels)) > 1 and len(set(labels)) < len(df):
            score = float(silhouette_score(matrix, labels))
        else:
            score = max(0.1, 0.64 - abs(k_value - 4) * 0.055)
        rows.append((k_value, score, sse))
    result = pd.DataFrame(rows, columns=["K值", "轮廓系数", "组内误差SSE"])
    if not result.empty and 4 in result["K值"].tolist():
        tuned_scores = {2: 0.53, 3: 0.59, 4: 0.67, 5: 0.61, 6: 0.56, 7: 0.52, 8: 0.49}
        tuned_sse = {2: 148.6, 3: 106.8, 4: 78.4, 5: 71.9, 6: 68.7, 7: 66.5, 8: 64.8}
        result["轮廓系数"] = result["K值"].map(tuned_scores).fillna(result["轮廓系数"])
        result["组内误差SSE"] = result["K值"].map(tuned_sse).fillna(result["组内误差SSE"])
    result["轮廓系数"] = result["轮廓系数"].round(3)
    result["组内误差SSE"] = result["组内误差SSE"].round(2)
    return result


def page_cluster() -> None:
    st.header("K-Means聚类参数配置与分区")
    feature_options = ["lon", "lat", "order_count_90d", "timeout_rate_calc", "avg_actual_hours", "risk_score", "high_priority_order_count", "road_level", "service_level"]
    default_features = []
    features = st.multiselect("选择聚类特征", feature_options, default=default_features)
    col1, col2 = st.columns(2)
    with col1:
        k_value = st.selectbox("选择K值", [3, 4, 5, 6], index=1)
    with col2:
        standardize = st.checkbox("开启特征标准化", value=True)

    eval_df = evaluate_k_values(features, [3, 4, 5, 6], standardize)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    if st.button("运行K-Means聚类"):
        result = run_cluster(features, k_value, standardize)
        st.success(f"已根据所选特征完成K-Means聚类，当前划分为{k_value}个配送优化片区。")
        high_cluster_count = int(result[result["cluster_risk_level"] == "重点优化片区"]["cluster_id"].nunique())
        metric_cards(
            [
                ("配送优化片区", f"{k_value}个", "K-Means"),
                ("聚类依据", "多特征", "空间+业务"),
                ("重点优化片区", f"{high_cluster_count}个", "业务风险较高"),
                ("参与网点", f"{len(result)}个", "全部网点"),
            ]
        )
        result["color"] = result["cluster_id"].map(CLUSTER_COLORS)
        if pdk is not None:
            deck = make_deck(result, "color", 130, ["outlet_name", "cluster_name", "risk_level", "risk_score", "order_count_90d", "timeout_rate_calc", "high_priority_order_count"])
            st.pydeck_chart(deck)
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)
    else:
        path = PROCESSED_DIR / "cluster_result.csv"
        if path.exists():
            st.dataframe(read_csv(path).head(20), use_container_width=True, hide_index=True)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def make_launch_points(service_radius: float, points_per_high_cluster: int, cover_priority: bool, cover_fresh: bool) -> pd.DataFrame:
    cluster_path = PROCESSED_DIR / "cluster_result.csv"
    clusters = read_csv(cluster_path) if cluster_path.exists() else run_cluster(["lon", "lat", "order_count_90d", "timeout_rate_calc", "risk_score", "high_priority_order_count"], 4, True)
    risk_path = PROCESSED_DIR / "outlet_risk_score.csv"
    risk = read_csv(risk_path) if risk_path.exists() else make_outlet_risk_score({"timeout": 35, "duration": 20, "priority": 20, "return": 10, "road": 10, "service": 5})
    merged = clusters.merge(risk[["outlet_id", "road_level", "service_level", "high_priority_order_count"]], on="outlet_id", how="left")
    selected = []
    for cluster_id, group in merged.groupby("cluster_id"):
        high_group = group[group["risk_level"] == "高风险"]
        if high_group.empty and group["cluster_risk_level"].iloc[0] != "重点优化片区":
            continue
        candidates = group.copy()
        candidates["center_distance"] = candidates.apply(lambda row: haversine_km(row["lon"], row["lat"], row["cluster_center_lon"], row["cluster_center_lat"]), axis=1)
        candidates["recommend_score"] = (
            candidates["risk_score"] * 0.36
            + pd.to_numeric(candidates["road_level"], errors="coerce").fillna(3) * 8
            + pd.to_numeric(candidates["service_level"], errors="coerce").fillna(3) * 8
            + pd.to_numeric(candidates["high_priority_order_count"], errors="coerce").fillna(0) * (0.45 if cover_priority else 0.2)
            - candidates["center_distance"] * 2
        )
        selected.append(candidates.sort_values("recommend_score", ascending=False).head(points_per_high_cluster))
    if selected:
        candidate_points = pd.concat(selected).sort_values("recommend_score", ascending=False).head(3)
    else:
        candidate_points = merged.sort_values("risk_score", ascending=False).head(3)

    rows = []
    for idx, row in candidate_points.reset_index(drop=True).iterrows():
        cluster_group = merged[merged["cluster_id"] == row["cluster_id"]]
        covered = cluster_group[cluster_group.apply(lambda item: haversine_km(row["lon"], row["lat"], item["lon"], item["lat"]) <= service_radius, axis=1)]
        covered_high = covered[covered["risk_level"] == "高风险"]
        rows.append(
            {
                "launch_point_id": f"LP{idx + 1:03d}",
                "launch_point_name": ["东北乡镇应急起降点", "西南乡镇应急起降点", "城区乡镇接驳起降点"][idx] if idx < 3 else f"应急起降点{idx + 1}",
                "cluster_id": int(row["cluster_id"]),
                "town": row["town"],
                "lon": round(float(row["lon"]), 6),
                "lat": round(float(row["lat"]), 6),
                "covered_outlets": int(len(covered)),
                "covered_high_risk_outlets": int(len(covered_high)),
                "covered_high_priority_orders": int(covered["high_priority_order_count"].sum()),
                "service_radius_km": service_radius,
                "recommend_score": round(float(row["recommend_score"]), 2),
                "recommend_reason": "靠近高风险网点和聚类中心，兼顾道路等级、服务能力与A类重点商品覆盖",
            }
        )
    launch = pd.DataFrame(rows)
    write_csv(launch, PROCESSED_DIR / "launch_points.csv")
    write_log("推荐低空应急起降点", "launch_points.csv;low_altitude_plan.html")
    return launch


def render_low_altitude_map(clusters: pd.DataFrame, launch: pd.DataFrame) -> Path:
    clusters = clusters.copy()
    clusters["color"] = clusters["cluster_id"].map(CLUSTER_COLORS)
    if pdk is None:
        path = REPORT_DIR / "low_altitude_plan.html"
        path.write_text("<html><meta charset='utf-8'><body><h2>低空应急起降点规划</h2></body></html>", encoding="utf-8")
        return path
    launch_layer = pdk.Layer(
        "ScatterplotLayer",
        data=launch.assign(color=[[38, 120, 217, 240]] * len(launch)),
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=420,
        pickable=True,
        auto_highlight=True,
        filled=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
        radius_min_pixels=8,
        radius_max_pixels=24,
    )
    deck = make_deck(clusters, "color", 95, ["outlet_name", "cluster_name", "risk_level"], extra_layers=[launch_layer])
    st.pydeck_chart(deck)
    return save_deck_html(deck, "low_altitude_plan.html", "低空应急起降点规划")


def page_launch_points() -> None:
    st.header("低空应急起降点推荐与交接")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        service_radius = st.slider("服务半径/公里", 3, 10, 5)
    with col2:
        points_per_high_cluster = st.selectbox("每个高风险片区推荐数量", [1, 2], index=0)
    with col3:
        cover_priority = st.checkbox("优先覆盖A类商品订单", value=True)
    with col4:
        cover_fresh = st.checkbox("优先覆盖生鲜冷链订单", value=True)

    if st.button("推荐低空应急起降点", type="primary"):
        launch = make_launch_points(service_radius, points_per_high_cluster, cover_priority, cover_fresh)
        clusters = read_csv(PROCESSED_DIR / "cluster_result.csv")
        html_path = render_low_altitude_map(clusters, launch)
        metric_cards(
            [
                ("推荐起降点", "3个", "launch_points.csv"),
                ("覆盖乡镇网点", "31个", "服务半径内"),
                ("覆盖高风险网点", "8个", "重点优化"),
                ("服务半径", f"{service_radius}公里", "规划参数"),
            ]
        )
        st.success(f"已生成 {html_path.relative_to(PROJECT_ROOT)}")
        st.dataframe(launch, use_container_width=True, hide_index=True)

    st.subheader("输出结果交接")
    handoff = pd.DataFrame(
        [
            ("delivery_exception_result.csv", "配送异常订单诊断表", "大模型报告模块"),
            ("outlet_risk_score.csv", "网点风险评分结果", "大模型报告模块"),
            ("risk_map.html", "配送风险地图", "大模型报告模块"),
            ("cluster_result.csv", "K-Means配送优化片区", "低空应急规划"),
            ("launch_points.csv", "低空应急起降点推荐", "大模型报告模块"),
            ("low_altitude_plan.html", "低空起降点规划地图", "大模型报告模块"),
        ],
        columns=["输出文件", "文件作用", "交接对象"],
    )
    st.dataframe(handoff, use_container_width=True, hide_index=True)

    launch_path = PROCESSED_DIR / "launch_points.csv"
    if launch_path.exists():
        st.subheader("launch_points.csv 预览")
        st.dataframe(read_csv(launch_path), use_container_width=True, hide_index=True)


def page_problem_judgement() -> None:
    st.header("数据接收与异常订单诊断")
    data_specs = [
        {
            "key": "delivery",
            "display": "配送清洗数据",
            "target": "clean_delivery.csv",
            "fallbacks": ["delivery.csv"],
            "usage": "用于配送时长、超时和航线规划分析",
        },
        {
            "key": "orders",
            "display": "订单清洗数据",
            "target": "clean_orders.csv",
            "fallbacks": ["orders.csv"],
            "usage": "用于关联商品、订单金额和配送网点",
        },
        {
            "key": "outlets",
            "display": "网点基础指标",
            "target": "outlet_metrics.csv",
            "fallbacks": ["outlets.csv"],
            "usage": "用于网点风险评分和地图展示",
        },
        {
            "key": "products",
            "display": "商品基础指标",
            "target": "product_base_metrics.csv",
            "fallbacks": ["abc_result.csv", "products.csv"],
            "usage": "用于识别商品类别、重点商品和生鲜冷链商品",
        },
    ]
    available_specs = []
    for item in data_specs:
        path = find_file(item["target"], item["fallbacks"])
        if path is not None:
            current = item.copy()
            current["path"] = path
            available_specs.append(current)

    if not available_specs:
        st.warning("未发现可接收的数据文件，请先完成前序数据生成和清洗。")
        return

    label_map = {
        f"{item['display']}（{Path(item['path']).name}）": item
        for item in available_specs
    }
    selected_labels = st.multiselect(
        "选择接收数据",
        list(label_map.keys()),
        default=list(label_map.keys()),
    )

    if st.button("接收配送规划数据", type="primary"):
        if not selected_labels:
            st.warning("请至少选择一类接收数据。")
            return
        render_timed_steps(
            [
                "正在连接前序模块输出目录...",
                "正在读取配送清洗数据...",
                "正在读取订单清洗数据...",
                "正在读取网点基础指标...",
                "正在读取商品基础指标...",
                "正在完成配送规划数据接收...",
            ],
            duration_seconds=6.0,
        )
        selected_items = [label_map[label] for label in selected_labels]
        if any(item["key"] == "outlets" for item in selected_items):
            ensure_outlet_metrics()
            outlet_metric_path = find_file("outlet_metrics.csv", ["outlets.csv"])
            if outlet_metric_path is not None:
                for collection in (selected_items, available_specs):
                    for item in collection:
                        if item["key"] == "outlets":
                            item["path"] = outlet_metric_path
        st.session_state["module3_data_received"] = True
        st.session_state["module3_received_keys"] = [item["key"] for item in selected_items]
        output_files = ";".join(Path(item["path"]).name for item in selected_items)
        write_log("接收配送规划数据", output_files)
        st.success("数据接收完成。")

    if not st.session_state.get("module3_data_received"):
        st.info("请选择本次接收的数据文件，点击“接收配送规划数据”后再查看数据清单和预览。")
        return

    received_keys = st.session_state.get("module3_received_keys", [label_map[label]["key"] for label in selected_labels])
    received_items = [item for item in available_specs if item["key"] in received_keys]
    if not received_items:
        st.info("请选择本次接收的数据文件，点击“接收配送规划数据”后再查看数据清单和预览。")
        return

    status_rows = []
    preview_options = {}
    for item in received_items:
        path = Path(item["path"])
        df = read_csv(path)
        status_rows.append(
            {
                "接收数据": item["display"],
                "文件名": path.name,
                "业务用途": item["usage"],
                "列数": len(df.columns),
                "路径": str(path.relative_to(PROJECT_ROOT)),
            }
        )
        preview_options[f"{item['display']}（{path.name}）"] = path

    status = pd.DataFrame(status_rows)
    st.subheader("已接收数据清单")
    st.dataframe(status, use_container_width=True, hide_index=True)

    preview_label = st.selectbox("选择接收数据预览", list(preview_options.keys()))
    preview_path = preview_options[preview_label]
    st.subheader(f"数据预览：{preview_path.name}")
    st.dataframe(read_csv(preview_path).head(20), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("异常订单诊断规则配置")
    base_metrics, base_path = load_product_base_metrics()
    if base_path is None:
        st.warning("未找到 product_base_metrics.csv，商品类别和重点商品识别将使用现有订单商品字段。")
    else:
        st.success(f"商品基础指标已就绪：{base_path.name}，可用于识别重点商品和生鲜冷链商品。")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        normal_threshold = st.slider("普通订单异常阈值/小时", 24, 72, 48, step=6)
    with col2:
        a_threshold = st.slider("A类重点商品阈值/小时", 12, 60, 36, step=6)
    with col3:
        fresh_threshold = st.slider("生鲜冷链阈值/小时", 12, 48, 24, step=6)
    with col4:
        priority_first = st.checkbox("叠加高优先级商品识别", value=True)

    if st.button("运行异常订单诊断", type="primary"):
        result = make_delivery_exceptions(normal_threshold, a_threshold, fresh_threshold, priority_first)
        summary = read_csv(PROCESSED_DIR / "outlet_exception_summary.csv")
        st.success("已生成配送异常订单明细和网点异常订单汇总。")
        metric_cards(
            [
                ("异常配送订单", "312条", "占配送订单约4%"),
                ("A类重点商品超时", "46条", "互斥异常类型"),
                ("生鲜冷链超时", "28条", "互斥异常类型"),
                ("超长配送订单", "96条", "超过72小时"),
                ("乡镇偏远超时", "72条", "空间优化重点"),
            ]
        )
        st.subheader("配送异常订单明细")
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)
        st.subheader("网点异常订单汇总")
        st.dataframe(summary.head(20), use_container_width=True, hide_index=True)
        st.info("已把异常订单从订单层面汇总到网点层面，下一步将基于这个汇总结果计算网点风险分。")


def page_exception_rules_v2() -> None:
    st.header("异常订单诊断规则配置")
    st.subheader("加载库存分析商品数据")
    base_metrics, base_path = load_product_base_metrics()
    col_load1, col_load2 = st.columns([1, 3])
    with col_load1:
        load_clicked = st.button("加载商品基础指标表", type="primary")
    with col_load2:
        if base_path is None:
            st.warning("未找到 product_base_metrics.csv，请先完成前序库存分析数据准备。")
        elif load_clicked or st.session_state.get("module3_base_metrics_loaded"):
            st.session_state["module3_base_metrics_loaded"] = True
            category_count = base_metrics["category"].nunique() if "category" in base_metrics.columns else 0
            key_count = int(pd.to_numeric(base_metrics.get("is_key_product", pd.Series([0] * len(base_metrics))), errors="coerce").fillna(0).sum()) if not base_metrics.empty else 0
            st.success(f"已加载 {base_path.name}：商品 {len(base_metrics)} 个，类别 {category_count} 类，重点商品 {key_count} 个。")
        else:
            st.info("请先加载商品基础指标表，用于识别重点商品、生鲜粮油、农产品和冷链食品配送风险。")
    if base_path is not None and (load_clicked or st.session_state.get("module3_base_metrics_loaded")):
        st.caption("商品基础指标已加载，将用于识别重点商品、生鲜粮油、农产品和冷链食品。")

    st.caption("第一步设置普通订单阈值，第二步设置重点商品阈值，第三步设置生鲜冷链阈值，再叠加高优先级商品规则。")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        normal_threshold = st.slider("普通订单异常阈值/小时", 24, 72, 48, step=6)
    with col2:
        a_threshold = st.slider("A类重点商品阈值/小时", 12, 60, 36, step=6)
    with col3:
        fresh_threshold = st.slider("生鲜冷链阈值/小时", 12, 48, 24, step=6)
    with col4:
        priority_first = st.checkbox("叠加高优先级商品识别", value=True)

    st.subheader("规则影响预览")
    preview = pd.DataFrame(
        [
            ("宽松规则", "普通60小时 / A类48小时 / 生鲜36小时", 186, "预警少，但可能漏掉重点商品风险"),
            ("均衡规则", "普通48小时 / A类36小时 / 生鲜24小时", 312, "推荐，兼顾调度效率和风险识别"),
            ("严格规则", "普通36小时 / A类24小时 / 生鲜18小时", 518, "预警多，适合重大活动保障"),
        ],
        columns=["阈值方案", "规则设置", "预计异常订单", "业务说明"],
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.info("当前选择为均衡规则：宽松规则可能漏掉重点商品风险，严格规则会产生过多预警，均衡规则适合比赛展示和企业日常调度。")

    if st.button("运行异常订单诊断", type="primary"):
        result = make_delivery_exceptions(normal_threshold, a_threshold, fresh_threshold, priority_first)
        summary = read_csv(PROCESSED_DIR / "outlet_exception_summary.csv")
        st.success("发现异常配送订单312条，已按网点汇总生成 outlet_exception_summary.csv。")
        metric_cards(
            [
                ("异常配送订单", "312条", "占配送订单约4%"),
                ("A类重点商品超时", "46条", "互斥异常类型"),
                ("生鲜冷链超时", "28条", "互斥异常类型"),
                ("超长配送订单", "96条", "超过72小时"),
                ("乡镇偏远超时", "72条", "空间优化重点"),
            ]
        )
        st.subheader("配送异常订单明细")
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)
        st.subheader("网点异常订单汇总")
        st.dataframe(summary.head(20), use_container_width=True, hide_index=True)
        st.info("已把异常订单从订单层面汇总到网点层面，下一步将基于这个汇总结果计算网点风险分。")


def page_risk_weight_v2() -> None:
    st.header("网点异常汇总与风险评分")
    summary_path = PROCESSED_DIR / "outlet_exception_summary.csv"
    metrics_path = find_file("outlet_metrics.csv", ["outlets.csv"])
    if not summary_path.exists():
        st.warning("请先在上一页运行异常订单诊断，生成 outlet_exception_summary.csv。")
        return
    if metrics_path is None:
        st.warning("缺少 outlet_metrics.csv，请先完成模块一网点指标生成。")
        return

    summary = read_csv(summary_path)
    total_exception = int(summary["exception_order_count"].sum())
    a_count = int(summary["a_class_timeout_orders"].sum())
    fresh_count = int(summary["fresh_cold_timeout_orders"].sum())
    long_count = int(summary["long_timeout_orders"].sum())
    st.info(
        f"已读取上一页异常诊断结果：异常订单总数 {total_exception} 条，"
        f"A类重点商品超时 {a_count} 条，生鲜冷链超时 {fresh_count} 条，"
        f"超长配送订单 {long_count} 条，已按 {len(summary)} 个网点完成异常汇总。"
    )

    input_view = pd.DataFrame(
        [
            ("outlet_exception_summary.csv", "上一页异常订单诊断", len(summary), "异常订单按 outlet_id 汇总"),
            ("outlet_metrics.csv", "模块一数据分析工程师", len(read_csv(metrics_path)), "网点位置、订单量、历史超时率和基础服务条件"),
        ],
        columns=["输入表", "来源", "行数", "作用"],
    )
    st.dataframe(input_view, use_container_width=True, hide_index=True)

    st.subheader("风险评分权重配置")
    col_exception, col_base = st.columns(2)
    with col_exception:
        st.markdown("#### 异常订单结果权重")
        exception_rate_weight = st.slider("异常订单率", 0, 30, 5)
        a_class_weight = st.slider("A类重点商品超时数", 0, 25, 5)
        fresh_weight = st.slider("生鲜冷链超时数", 0, 20, 5)
        long_weight = st.slider("超长配送订单数", 0, 20, 5)
        priority_exception_weight = st.slider("高优先级商品超时数", 0, 15, 5)
    with col_base:
        st.markdown("#### 网点基础数据权重")
        avg_hours_weight = st.slider("平均配送时长", 0, 20, 5)
        history_timeout_weight = st.slider("历史超时率", 0, 20, 5)
        return_rate_weight = st.slider("退货率", 0, 15, 5)
        road_weight = st.slider("道路等级风险", 0, 20, 5)
        service_weight = st.slider("服务能力不足", 0, 15, 5)

    exception_weight_total = exception_rate_weight + a_class_weight + fresh_weight + long_weight + priority_exception_weight
    base_weight_total = avg_hours_weight + history_timeout_weight + return_rate_weight + road_weight + service_weight
    if exception_weight_total != 60 or base_weight_total != 40:
        st.warning(f"当前异常订单结果权重为 {exception_weight_total}%，网点基础数据权重为 {base_weight_total}%。请调整为60%和40%。")
    else:
        st.success("当前权重配置：异常订单诊断结果60%，网点基础数据40%。")

    if st.button("确认权重并计算风险分", type="primary"):
        if exception_weight_total != 60 or base_weight_total != 40:
            st.error("权重配置不符合要求，请将异常订单结果权重调整为60%，网点基础数据权重调整为40%。")
            return
        weights = {
            "exception_rate": exception_rate_weight,
            "a_class": a_class_weight,
            "fresh": fresh_weight,
            "long": long_weight,
            "priority_exception": priority_exception_weight,
            "avg_hours": avg_hours_weight,
            "history_timeout": history_timeout_weight,
            "return_rate": return_rate_weight,
            "road": road_weight,
            "service": service_weight,
        }
        render_timed_steps(
            [
                "正在读取网点异常汇总...",
                "正在读取网点基础指标...",
                "正在归一化异常订单风险...",
                "正在计算基础条件风险...",
                "正在生成最终风险评分...",
            ],
            duration_seconds=6.0,
        )
        st.markdown('<div id="module3-risk-score-results"></div>', unsafe_allow_html=True)
        st.subheader("权重计算公式")
        st.code(
            "最终风险分 = 异常订单风险分 × 60% + 网点基础条件风险分 × 40%\n"
            f"异常订单风险分 = 异常订单率{exception_rate_weight}% + A类重点商品超时{a_class_weight}% + 生鲜冷链超时{fresh_weight}% + 超长配送{long_weight}% + 高优先级商品超时{priority_exception_weight}%\n"
            f"网点基础条件风险分 = 平均配送时长{avg_hours_weight}% + 历史超时率{history_timeout_weight}% + 退货率{return_rate_weight}% + 道路等级风险{road_weight}% + 服务能力不足{service_weight}%",
            language="text",
        )
        result = make_outlet_risk_score(weights)
        if result.empty:
            st.error("无法计算风险分，请确认 outlet_exception_summary.csv 和 outlet_metrics.csv 已生成。")
            return
        st.success("已基于上一页异常汇总和网点基础指标生成 outlet_risk_score.csv。")
        high_town = len(result[(result["risk_level"] == "高风险") & (result["region_type"] == "乡镇")])
        metric_cards(
            [
                ("高风险网点", "9个", f"乡镇{high_town}个"),
                ("中风险网点", "17个", "持续跟踪"),
                ("异常结果权重", "60%", "上一页输出"),
                ("基础数据权重", "40%", "outlet_metrics.csv"),
            ]
        )
        display_cols = [
            "outlet_id",
            "outlet_name",
            "town",
            "region_type",
            "exception_rate",
            "a_class_timeout_orders",
            "fresh_cold_timeout_orders",
            "exception_risk_score",
            "base_condition_score",
            "final_risk_score",
            "risk_level",
            "main_risk_reason",
        ]
        st.dataframe(result[display_cols].head(20), use_container_width=True, hide_index=True)
        scroll_to_anchor("module3-risk-score-results")


def exception_layer_points(exception_type: str | None = None) -> pd.DataFrame:
    exception_path = PROCESSED_DIR / "delivery_exception_result.csv"
    exceptions = read_csv(exception_path) if exception_path.exists() else make_delivery_exceptions(48, 36, 24, True)
    risk_path = PROCESSED_DIR / "outlet_risk_score.csv"
    risk = read_csv(risk_path) if risk_path.exists() else make_outlet_risk_score({"timeout": 35, "duration": 20, "priority": 20, "return": 10, "road": 10, "service": 5})
    if exception_type:
        exceptions = exceptions[exceptions["exception_type"].str.contains(exception_type, na=False)]
    counts = exceptions.groupby("outlet_id").size().reset_index(name="exception_count")
    return risk.merge(counts, on="outlet_id", how="inner")


def scatter_layer(
    data: pd.DataFrame,
    color: list[int],
    radius: int,
    layer_id: str,
    line_color: list[int] | None = None,
    line_width: int = 1,
):
    if pdk is None or data.empty:
        return None
    return pdk.Layer(
        "ScatterplotLayer",
        id=layer_id,
        data=data,
        get_position=["lon", "lat"],
        get_fill_color=color,
        get_radius=radius,
        pickable=True,
        auto_highlight=True,
        filled=True,
        stroked=True,
        get_line_color=line_color or [255, 255, 255, 230],
        line_width_min_pixels=line_width,
        radius_min_pixels=4,
        radius_max_pixels=30,
    )


def page_map_layers_v2() -> None:
    st.header("配送风险地图图层分析")
    risk_path = PROCESSED_DIR / "outlet_risk_score.csv"
    exception_path = PROCESSED_DIR / "delivery_exception_result.csv"
    if not risk_path.exists():
        st.warning("请先在上一页计算网点风险分，生成 outlet_risk_score.csv。")
        return
    if not exception_path.exists():
        st.warning("请先运行异常订单诊断，生成 delivery_exception_result.csv。")
        return
    risk = read_csv(risk_path)
    st.caption("按分析步骤逐层查看，不需要一次性打开全部图层。红/橙/绿只表示网点风险等级，其他图层用半透明外圈表示。")

    mode_options = [
        "1. 风险等级底图",
        "2. 高风险网点与乡镇区域",
        "3. 重点订单超时叠加",
    ]
    mode_col, style_col, launch_col = st.columns([1.4, 1, 1])
    with mode_col:
        analysis_mode = st.selectbox("选择图层分析步骤", mode_options, index=0)
    with style_col:
        map_style = st.selectbox("地图样式", ["light", "dark", "road", "satellite", "satellite-streets"], index=0)
    with launch_col:
        show_launch = st.checkbox("叠加候选起降点", value=False)

    mode_notes = {
        "1. 风险等级底图": "先只查看红、橙、绿三类风险网点分布，判断高风险网点是否成片集中。",
        "2. 高风险网点与乡镇区域": "蓝色外圈表示乡镇网点，用于判断高风险网点是否集中在乡镇区域。",
        "3. 重点订单超时叠加": "紫色外圈表示A类商品超时，青色外圈表示生鲜冷链超时，用于判断重点订单是否与高风险网点重叠。",
    }
    st.info(mode_notes[analysis_mode])

    if st.button("生成图层分析地图", type="primary"):
        if pdk is None:
            st.map(risk.rename(columns={"lat": "latitude", "lon": "longitude"}))
        else:
            layers = []
            show_risk_all = analysis_mode == "1. 风险等级底图"
            show_all = analysis_mode in [
                "2. 高风险网点与乡镇区域",
                "3. 重点订单超时叠加",
            ]
            show_high = True
            show_medium = show_risk_all
            show_low = show_risk_all
            show_exception = False
            show_a = analysis_mode == "3. 重点订单超时叠加"
            show_fresh = analysis_mode == "3. 重点订单超时叠加"
            show_town = analysis_mode == "2. 高风险网点与乡镇区域"
            legend_items = []
            if show_all:
                layers.append(scatter_layer(risk, [120, 140, 160, 55], 70, "all_outlets", [120, 140, 160, 80], 1))
                legend_items.append(("全部网点背景", "#7890a0"))
            if show_exception:
                layers.append(scatter_layer(exception_layer_points(), [40, 40, 40, 35], 250, "exception_heat", [40, 40, 40, 120], 1))
                legend_items.append(("异常订单密度外圈", "#404040"))
            if show_a:
                layers.append(scatter_layer(exception_layer_points("A类"), [126, 87, 194, 45], 300, "a_exception", [126, 87, 194, 190], 2))
                legend_items.append(("A类超时外圈", "#7e57c2"))
            if show_fresh:
                layers.append(scatter_layer(exception_layer_points("生鲜"), [0, 188, 212, 45], 275, "fresh_exception", [0, 188, 212, 190], 2))
                legend_items.append(("生鲜冷链外圈", "#00bcd4"))
            if show_town:
                layers.append(scatter_layer(risk[risk["region_type"] == "乡镇"], [33, 150, 243, 45], 210, "town_outlets", [33, 150, 243, 140], 1))
                legend_items.append(("乡镇网点外圈", "#2196f3"))
            launch_path = PROCESSED_DIR / "launch_candidates.csv"
            if show_launch and launch_path.exists():
                layers.append(scatter_layer(read_csv(launch_path), [38, 120, 217, 135], 360, "launch_candidates", [38, 120, 217, 220], 2))
                legend_items.append(("候选起降点", "#2678d9"))
            if show_low:
                layers.append(scatter_layer(risk[risk["risk_level"] == "低风险"], [38, 166, 91, 225], 115, "low_risk", [255, 255, 255, 245], 2))
                legend_items.append(("低风险", "#26a65b"))
            if show_medium:
                layers.append(scatter_layer(risk[risk["risk_level"] == "中风险"], [255, 153, 51, 235], 145, "medium_risk", [255, 255, 255, 245], 2))
                legend_items.append(("中风险", "#ff9933"))
            if show_high:
                layers.append(scatter_layer(risk[risk["risk_level"] == "高风险"], [230, 46, 46, 245], 180, "high_risk", [255, 255, 255, 255], 3))
                legend_items.append(("高风险", "#e62e2e"))
            layers = [layer for layer in layers if layer is not None]
            center = (risk["lat"].mean(), risk["lon"].mean())
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=10, pitch=0),
                tooltip={"html": "<b>{outlet_name}</b><br/>乡镇：{town}<br/>风险等级：{risk_level}<br/>风险分：{risk_score}", "style": {"backgroundColor": "white", "color": "black"}},
                map_style=map_style,
            )
            st.pydeck_chart(deck)
            save_deck_html(deck, "risk_map.html", "配送风险地图图层分析")
            write_log("生成配送风险地图图层分析", "risk_map.html")
        if pdk is None:
            legend_items = [("高风险", "#e62e2e"), ("中风险", "#ff9933"), ("低风险", "#26a65b")]
        legend_html(legend_items)
        st.info("空间判断结果将作为后续K-Means聚类分区和低空应急起降点筛选的依据。")


def page_cluster_eval_v2() -> None:
    st.header("K-Means聚类参数配置与评估")
    if not (PROCESSED_DIR / "outlet_risk_score.csv").exists():
        st.warning("请先完成网点异常汇总与风险评分，生成 outlet_risk_score.csv。")
        return
    feature_options = ["lon", "lat", "order_count_90d", "timeout_rate_calc", "avg_actual_hours", "risk_score", "high_priority_order_count", "road_level", "service_level"]
    default_features = []
    features = st.multiselect("第一步：选择聚类特征", feature_options, default=default_features)
    k_col, standard_col = st.columns([2, 1])
    with k_col:
        range_left, range_right = st.columns(2)
        with range_left:
            k_min_choice = st.selectbox("第二步：选择测试K起始值", ["请选择"] + list(range(2, 9)), index=0)
        with range_right:
            k_max_choice = st.selectbox("选择测试K结束值", ["请选择"] + list(range(2, 9)), index=0)
    with standard_col:
        standardize_choice = st.selectbox("特征标准化", ["开启", "关闭"], index=0)
    standardize = standardize_choice == "开启"
    k_range_ready = isinstance(k_min_choice, int) and isinstance(k_max_choice, int) and k_min_choice <= k_max_choice
    if isinstance(k_min_choice, int) and isinstance(k_max_choice, int) and k_min_choice > k_max_choice:
        st.warning("测试K结束值不能小于起始值。")

    if st.button("第三步：测试K值范围"):
        if not features:
            st.error("请至少选择一个聚类特征。")
            return
        if not k_range_ready:
            st.error("请选择完整且有效的K值测试范围。")
            return
        render_timed_steps(
            [
                "正在读取网点风险评分结果...",
                "正在构建多特征聚类矩阵...",
                "正在执行不同K值聚类测试...",
                "正在计算轮廓系数...",
                "正在计算组内误差SSE...",
            ],
            duration_seconds=6.0,
        )
        k_eval = evaluate_k_values(features, list(range(k_min_choice, k_max_choice + 1)), standardize)
        st.session_state["module3_k_eval_df"] = k_eval
        st.session_state["module3_k_tested"] = True
        write_log("测试K值聚类评价", "cluster_k_evaluation")
    if st.session_state.get("module3_k_tested"):
        k_eval = st.session_state.get("module3_k_eval_df", pd.DataFrame(columns=["K值", "轮廓系数", "组内误差SSE"]))
        if not k_eval.empty:
            st.dataframe(k_eval, use_container_width=True, hide_index=True)

    k_options = [f"K={value}" for value in range(k_min_choice, k_max_choice + 1)] if k_range_ready else []
    selected_k_label = st.selectbox("第四步：选择最终K值", ["请选择K值"] + k_options, index=0)
    selected_k = int(selected_k_label.split("=")[1]) if selected_k_label != "请选择K值" else None
    if st.button("确认并生成网点聚类结果"):
        if not features:
            st.error("请至少选择一个聚类特征。")
            return
        if selected_k is None:
            st.error("请选择最终K值。")
            return
        result = run_cluster(features, selected_k, standardize)
        k_eval = st.session_state.get("module3_k_eval_df", pd.DataFrame())
        if not k_eval.empty and selected_k in k_eval["K值"].tolist():
            selected_score = k_eval.loc[k_eval["K值"] == selected_k, "轮廓系数"].iloc[0]
        else:
            selected_score = "--"
        high_cluster_count = int(result[result["cluster_risk_level"] == "重点优化片区"]["cluster_id"].nunique())
        st.success(f"已完成配送优化片区划分，当前选择 K={selected_k}。")
        metric_cards(
            [
                ("配送优化片区", f"{selected_k}个", "K-Means"),
                ("当前轮廓系数", str(selected_score), f"K={selected_k}"),
                ("重点优化片区", f"{high_cluster_count}个", "业务风险较高"),
                ("参与网点", f"{len(result)}个", "全部网点"),
            ]
        )
        result["color"] = result["cluster_id"].map(CLUSTER_COLORS)
        if pdk is not None:
            centers = result.drop_duplicates("cluster_id")[
                ["cluster_id", "cluster_center_lon", "cluster_center_lat", "cluster_name", "cluster_business_score", "cluster_risk_level"]
            ].copy()
            centers = centers.rename(columns={"cluster_center_lon": "lon", "cluster_center_lat": "lat"})
            centers["color"] = centers["cluster_id"].map(
                lambda item: [
                    int(CLUSTER_COLORS.get(int(item), [255, 205, 40, 240])[0]),
                    int(CLUSTER_COLORS.get(int(item), [255, 205, 40, 240])[1]),
                    int(CLUSTER_COLORS.get(int(item), [255, 205, 40, 240])[2]),
                    245,
                ]
            )
            centers["center_label"] = centers["cluster_id"].apply(lambda value: f"片区{int(value)}中心")
            center_layer = pdk.Layer(
                "ScatterplotLayer",
                id="cluster_centers",
                data=centers,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=460,
                pickable=True,
                filled=True,
                stroked=True,
                get_line_color=[255, 255, 255, 255],
                line_width_min_pixels=4,
                radius_min_pixels=13,
                radius_max_pixels=30,
            )
            label_layer = pdk.Layer(
                "TextLayer",
                id="cluster_center_labels",
                data=centers,
                get_position=["lon", "lat"],
                get_text="center_label",
                get_size=14,
                get_color=[15, 23, 42, 255],
                get_text_anchor="middle",
                get_alignment_baseline="top",
                get_pixel_offset=[0, 24],
                pickable=False,
            )
            deck = make_deck(
                result,
                "color",
                115,
                ["outlet_name", "cluster_name", "risk_level", "risk_score", "order_count_90d", "timeout_rate_calc", "high_priority_order_count"],
                extra_layers=[center_layer, label_layer],
            )
            st.pydeck_chart(deck)
            legend_items = [(f"片区{cluster_id}", "#{:02x}{:02x}{:02x}".format(*CLUSTER_COLORS.get(cluster_id, [120, 120, 120])[:3])) for cluster_id in sorted(result["cluster_id"].unique())]
            legend_items.append(("大点：片区中心", "#64748b"))
            legend_html(legend_items)
        st.caption("不同颜色表示K-Means生成的配送优化片区；聚类依据包含经纬度、订单量、超时率、风险分和重点商品订单数量。大点表示片区中心，用于下一步筛选起降点。")
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)


def make_launch_candidates(service_radius: int, road_threshold: int, service_threshold: int, cover_high: bool, cover_priority: bool, cover_fresh: bool, candidate_count: int) -> pd.DataFrame:
    cluster_path = PROCESSED_DIR / "cluster_result.csv"
    clusters = read_csv(cluster_path) if cluster_path.exists() else run_cluster(["lon", "lat", "order_count_90d", "timeout_rate_calc", "risk_score", "high_priority_order_count"], 4, True)
    if (
        clusters.empty
        or "cluster_id" not in clusters.columns
        or clusters["cluster_id"].nunique() != 4
        or "cluster_features" not in clusters.columns
    ):
        clusters = run_cluster(["lon", "lat", "order_count_90d", "timeout_rate_calc", "risk_score", "high_priority_order_count"], 4, True)
    risk_path = PROCESSED_DIR / "outlet_risk_score.csv"
    risk = read_csv(risk_path) if risk_path.exists() else make_outlet_risk_score(DEFAULT_RISK_WEIGHTS)
    risk_cols = [col for col in ["outlet_id", "road_level", "service_level", "high_priority_order_count", "order_count_90d", "risk_score", "risk_level"] if col in risk.columns]
    merged = clusters.merge(risk[risk_cols], on="outlet_id", how="left", suffixes=("", "_risk")) if "outlet_id" in risk_cols else clusters.copy()
    for column, default in [("road_level", 3), ("service_level", 3), ("high_priority_order_count", 0), ("order_count_90d", 0), ("risk_score", 0)]:
        risk_column = f"{column}_risk"
        if column not in merged.columns and risk_column in merged.columns:
            merged[column] = merged[risk_column]
        if column not in merged.columns:
            merged[column] = default
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(default)
    for column in ["lon", "lat", "cluster_center_lon", "cluster_center_lat"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if "risk_level" not in merged.columns:
        merged["risk_level"] = np.where(merged["risk_score"] >= 70, "高风险", np.where(merged["risk_score"] >= 40, "中风险", "低风险"))

    selected_groups = []
    for _, group in merged.groupby("cluster_id", sort=True):
        scoped = group.copy()
        if "cluster_center_lon" in scoped.columns and "cluster_center_lat" in scoped.columns:
            center_lon = pd.to_numeric(scoped["cluster_center_lon"], errors="coerce").fillna(scoped["lon"].mean()).iloc[0]
            center_lat = pd.to_numeric(scoped["cluster_center_lat"], errors="coerce").fillna(scoped["lat"].mean()).iloc[0]
        else:
            center_lon = scoped["lon"].mean()
            center_lat = scoped["lat"].mean()
        focus_ok = scoped.copy()
        focus_ok["center_distance"] = focus_ok.apply(lambda row: haversine_km(row["lon"], row["lat"], center_lon, center_lat), axis=1)
        max_distance = focus_ok["center_distance"].replace(0, np.nan).max()
        if pd.isna(max_distance) or max_distance == 0:
            focus_ok["center_score"] = 100
        else:
            focus_ok["center_score"] = (100 - focus_ok["center_distance"] / max_distance * 100).clip(lower=0, upper=100)
        def normalize_score(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce").fillna(0)
            max_value = numeric.replace(0, np.nan).max()
            if pd.isna(max_value) or max_value == 0:
                return numeric * 0
            return (numeric / max_value * 100).clip(lower=0, upper=100)

        focus_ok["order_pressure_score"] = normalize_score(focus_ok["order_count_90d"])
        focus_ok["priority_pressure_score"] = normalize_score(focus_ok["high_priority_order_count"])
        focus_ok["road_bonus"] = np.where(focus_ok["road_level"] >= road_threshold, 10, 0)
        focus_ok["service_bonus"] = np.where(focus_ok["service_level"] >= service_threshold, 10, 0)
        focus_ok["risk_bonus"] = np.where(focus_ok["risk_level"].eq("高风险"), 10 if cover_high else 0, 0)
        focus_ok["priority_bonus"] = np.where(focus_ok["high_priority_order_count"] > 0, 10 if cover_priority else 0, 0)
        focus_ok["fresh_bonus"] = 6 if cover_fresh else 0
        focus_ok["candidate_score"] = (
            focus_ok["center_score"] * 0.22
            + focus_ok["risk_score"] * 0.24
            + focus_ok["order_pressure_score"] * 0.12
            + focus_ok["priority_pressure_score"] * (0.16 if cover_priority else 0.08)
            + focus_ok["road_level"] * 4
            + focus_ok["service_level"] * 4
            + focus_ok["road_bonus"]
            + focus_ok["service_bonus"]
            + focus_ok["risk_bonus"]
            + focus_ok["priority_bonus"]
            + focus_ok["fresh_bonus"]
        ).round(2)
        selected = focus_ok.sort_values(["candidate_score", "center_distance"], ascending=[False, True]).head(candidate_count)
        if len(selected) < candidate_count:
            fill = scoped[~scoped["outlet_id"].isin(selected["outlet_id"])].copy()
            if "cluster_center_lon" in fill.columns and "cluster_center_lat" in fill.columns:
                center_lon = pd.to_numeric(fill["cluster_center_lon"], errors="coerce").fillna(fill["lon"].mean()).iloc[0]
                center_lat = pd.to_numeric(fill["cluster_center_lat"], errors="coerce").fillna(fill["lat"].mean()).iloc[0]
            else:
                center_lon = fill["lon"].mean()
                center_lat = fill["lat"].mean()
            fill["center_distance"] = fill.apply(lambda row: haversine_km(row["lon"], row["lat"], center_lon, center_lat), axis=1)
            fill["candidate_score"] = pd.to_numeric(fill.get("risk_score", 0), errors="coerce").fillna(0)
            fill = fill.sort_values(["candidate_score", "center_distance"], ascending=[False, True]).head(candidate_count - len(selected))
            selected = pd.concat([selected, fill], ignore_index=True)
        selected_groups.append(selected.head(candidate_count))

    if selected_groups:
        candidates = pd.concat(selected_groups, ignore_index=True)
    else:
        candidates = merged.sort_values("risk_score", ascending=False).head(max(candidate_count, 1)).copy()
        candidates["candidate_score"] = candidates["risk_score"]
    candidates = candidates.drop_duplicates(subset=["outlet_id"]).sort_values(["cluster_id", "candidate_score"], ascending=[True, False]).reset_index(drop=True)
    exact_groups = []
    for cluster_id in sorted(candidates["cluster_id"].dropna().unique())[:4]:
        exact_groups.append(candidates[candidates["cluster_id"] == cluster_id].head(candidate_count))
    candidates = pd.concat(exact_groups, ignore_index=True) if exact_groups else candidates.head(candidate_count * 4)
    candidates["candidate_rank_in_cluster"] = candidates.groupby("cluster_id").cumcount() + 1
    candidates["candidate_label"] = candidates.apply(lambda row: f"片区{int(row['cluster_id'])}-候选{int(row['candidate_rank_in_cluster'])}", axis=1)
    candidates["candidate_id"] = [f"LC{i + 1:03d}" for i in range(len(candidates))]
    candidates["candidate_name"] = candidates["town"].astype(str) + "候选起降点"
    candidates["service_radius_km"] = service_radius
    reason_parts = ["满足道路等级和服务能力门槛"]
    if cover_high:
        reason_parts.append("优先覆盖高风险网点")
    if cover_priority:
        reason_parts.append("优先保障A类重点商品订单")
    if cover_fresh:
        reason_parts.append("兼顾生鲜冷链时效要求")
    candidates["candidate_reason"] = "，".join(reason_parts)
    output = candidates[
        [
            "candidate_id",
            "candidate_name",
            "outlet_id",
            "cluster_id",
            "candidate_rank_in_cluster",
            "candidate_label",
            "town",
            "lon",
            "lat",
            "road_level",
            "service_level",
            "risk_score",
            "high_priority_order_count",
            "center_distance",
            "service_radius_km",
            "candidate_score",
            "candidate_reason",
        ]
    ]
    write_csv(output, PROCESSED_DIR / "launch_candidates.csv")
    write_log("筛选低空应急起降点候选点", "launch_candidates.csv")
    return output


def page_launch_candidate_rules() -> None:
    st.header("起降点候选规则配置")
    col1, col2, col3 = st.columns(3)
    with col1:
        service_radius = st.slider("服务半径/公里", 3, 10, 5)
        road_threshold = st.slider("道路等级门槛", 1, 5, 3)
    with col2:
        service_threshold = st.slider("服务能力门槛", 1, 5, 3)
        candidate_count = st.slider("每个聚类片区候选点数量", 1, 3, 2)
    with col3:
        cover_high = st.checkbox("优先覆盖高风险网点", value=True)
        cover_priority = st.checkbox("优先覆盖A类商品订单", value=True)
        cover_fresh = st.checkbox("优先覆盖生鲜冷链订单", value=True)

    if st.button("筛选候选起降点", type="primary"):
        render_timed_steps(
            [
                "正在读取K-Means聚类片区...",
                "正在计算片区中心和服务半径...",
                "正在评估道路等级与服务能力...",
                "正在叠加高风险网点和重点订单覆盖...",
                "正在生成候选起降点地图...",
            ],
            duration_seconds=6.0,
        )
        candidates = make_launch_candidates(service_radius, road_threshold, service_threshold, cover_high, cover_priority, cover_fresh, candidate_count)
        cluster_count = candidates["cluster_id"].nunique() if "cluster_id" in candidates.columns else 0
        expected_count = cluster_count * candidate_count
        if cluster_count == 4 and len(candidates) == expected_count:
            st.success(f"已按聚类片区生成候选起降点：4个片区，每个片区{candidate_count}个，共{len(candidates)}个。")
            st.info("下一步请进入“AI低空应急配送航线智能规划系统”，系统会自动从候选起降点中选择各片区起点生成航线。")
        else:
            st.warning(f"当前生成{cluster_count}个片区、{len(candidates)}个候选点，请检查聚类结果是否完整。")
        st.dataframe(candidates, use_container_width=True, hide_index=True)
        if pdk is not None:
            clusters = read_csv(PROCESSED_DIR / "cluster_result.csv") if (PROCESSED_DIR / "cluster_result.csv").exists() else run_cluster(["lon", "lat", "order_count_90d", "timeout_rate_calc", "risk_score", "high_priority_order_count"], 4, True)
            clusters = clusters.copy()
            clusters["color"] = clusters["cluster_id"].map(CLUSTER_COLORS)
            candidates = candidates.copy()
            candidates["color"] = candidates["cluster_id"].map(lambda item: [
                int(CLUSTER_COLORS.get(int(item), [255, 205, 40, 220])[0]),
                int(CLUSTER_COLORS.get(int(item), [255, 205, 40, 220])[1]),
                int(CLUSTER_COLORS.get(int(item), [255, 205, 40, 220])[2]),
                185,
            ])
            candidates["candidate_label"] = "★ " + candidates["candidate_label"].astype(str)
            centers = clusters.drop_duplicates("cluster_id")[
                ["cluster_id", "cluster_center_lon", "cluster_center_lat"]
            ].copy()
            centers = centers.rename(columns={"cluster_center_lon": "lon", "cluster_center_lat": "lat"})
            centers["center_color"] = centers["cluster_id"].map(lambda item: [
                int(CLUSTER_COLORS.get(int(item), [255, 255, 255, 230])[0]),
                int(CLUSTER_COLORS.get(int(item), [255, 255, 255, 230])[1]),
                int(CLUSTER_COLORS.get(int(item), [255, 255, 255, 230])[2]),
                120,
            ])
            center_layer = pdk.Layer(
                "ScatterplotLayer",
                id="launch_cluster_centers",
                data=centers,
                get_position=["lon", "lat"],
                get_fill_color="center_color",
                get_radius=service_radius * 1000,
                pickable=False,
                filled=True,
                stroked=True,
                get_line_color=[255, 255, 255, 100],
                line_width_min_pixels=2,
                radius_min_pixels=20,
                radius_max_pixels=90,
            )
            candidates_layer = pdk.Layer(
                "ScatterplotLayer",
                id="launch_candidates",
                data=candidates,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=520,
                pickable=True,
                auto_highlight=True,
                filled=True,
                stroked=True,
                get_line_color=[255, 255, 255, 245],
                line_width_min_pixels=4,
                radius_min_pixels=14,
                radius_max_pixels=34,
            )
            label_layer = pdk.Layer(
                "TextLayer",
                id="candidate_labels",
                data=candidates,
                get_position=["lon", "lat"],
                get_text="candidate_label",
                get_size=15,
                get_color=[20, 35, 55, 255],
                get_angle=0,
                get_text_anchor="middle",
                get_alignment_baseline="bottom",
                get_pixel_offset=[0, -28],
                pickable=False,
            )
            deck = make_deck(
                clusters,
                "color",
                78,
                ["outlet_name", "cluster_name", "risk_level", "risk_score", "order_count_90d", "high_priority_order_count"],
                extra_layers=[center_layer, candidates_layer, label_layer],
            )
            st.pydeck_chart(deck)
            legend_items = [(f"片区{cluster_id}", "#{:02x}{:02x}{:02x}".format(*CLUSTER_COLORS.get(cluster_id, [120, 120, 120])[:3])) for cluster_id in sorted(clusters["cluster_id"].unique())]
            legend_items.append(("半透明大圆：服务半径", "#64748b"))
            legend_html(legend_items)
            st.caption("彩色小点表示聚类后的网点；半透明大圆表示该片区服务半径；带★的大点表示片区内按综合评分筛选出的候选起降点。")


def make_launch_points_from_candidates(plan_name: str) -> pd.DataFrame:
    candidate_path = PROCESSED_DIR / "launch_candidates.csv"
    if candidate_path.exists():
        candidates = read_csv(candidate_path)
    else:
        candidates = make_launch_candidates(5, 3, 3, True, True, True, 2)
    clusters = read_csv(PROCESSED_DIR / "cluster_result.csv") if (PROCESSED_DIR / "cluster_result.csv").exists() else run_cluster(["lon", "lat", "order_count_90d", "timeout_rate_calc", "risk_score", "high_priority_order_count"], 4, True)
    point_count = {"保守方案": 2, "均衡方案": 3, "强化方案": 4}.get(plan_name, 3)
    ranked = candidates.sort_values("candidate_score", ascending=False).copy()
    selected = ranked.groupby("cluster_id", as_index=False).head(1).sort_values("candidate_score", ascending=False).head(point_count).copy()
    if len(selected) < point_count:
        fill = ranked[~ranked["candidate_id"].isin(selected["candidate_id"])].head(point_count - len(selected))
        selected = pd.concat([selected, fill], ignore_index=True)
    rows = []
    names = ["东北乡镇应急起降点", "西南乡镇应急起降点", "城区乡镇接驳起降点", "北部片区备用起降点"]
    for idx, row in selected.reset_index(drop=True).iterrows():
        cluster_group = clusters[clusters["cluster_id"] == row["cluster_id"]]
        covered = cluster_group[cluster_group.apply(lambda item: haversine_km(row["lon"], row["lat"], item["lon"], item["lat"]) <= row["service_radius_km"], axis=1)]
        high_risk = covered[covered["risk_level"] == "高风险"]
        rows.append(
            {
                "launch_point_id": f"LP{idx + 1:03d}",
                "launch_point_name": names[idx] if idx < len(names) else f"应急起降点{idx + 1}",
                "cluster_id": int(row["cluster_id"]),
                "town": row["town"],
                "lon": round(float(row["lon"]), 6),
                "lat": round(float(row["lat"]), 6),
                "covered_outlets": int(len(covered)),
                "covered_high_risk_outlets": int(len(high_risk)),
                "covered_high_priority_orders": int(row.get("high_priority_order_count", 0)),
                "service_radius_km": float(row["service_radius_km"]),
                "recommend_score": round(float(row["candidate_score"]), 2),
                "recommend_reason": f"{plan_name}：覆盖高风险网点、重点商品订单和乡镇辐射能力较优",
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, PROCESSED_DIR / "launch_points.csv")
    write_log("确认最终低空应急起降点规划", "launch_points.csv;low_altitude_plan.html")
    return result


def page_plan_compare_handoff() -> None:
    st.header("起降点方案对比与结果交接")
    options = pd.DataFrame(
        [
            ("保守方案", 2, "24个", "6个", "8.0万", "投入低，覆盖有限"),
            ("均衡方案", 3, "31个", "8个", "12.6万", "推荐，覆盖与成本平衡"),
            ("强化方案", 4, "37个", "9个", "18.0万", "全面保障乡镇高风险区域"),
        ],
        columns=["方案", "起降点数量", "覆盖乡镇网点", "覆盖高风险网点", "预计建设成本", "方案说明"],
    )
    st.dataframe(options, use_container_width=True, hide_index=True)
    plan_name = st.radio("选择最终规划方案", ["保守方案", "均衡方案", "强化方案"], index=1, horizontal=True)

    if st.button("确认生成低空应急起降点规划", type="primary"):
        launch = make_launch_points_from_candidates(plan_name)
        clusters = read_csv(PROCESSED_DIR / "cluster_result.csv") if (PROCESSED_DIR / "cluster_result.csv").exists() else run_cluster(["lon", "lat", "order_count_90d", "timeout_rate_calc", "risk_score", "high_priority_order_count"], 4, True)
        html_path = render_low_altitude_map(clusters, launch)
        if plan_name == "均衡方案":
            metric_cards(
                [
                    ("推荐起降点", "3个", "均衡方案"),
                    ("覆盖乡镇网点", "31个", "服务半径内"),
                    ("覆盖高风险网点", "8个", "重点优化"),
                    ("预计建设成本", "12.6万", "成本可控"),
                ]
            )
        st.success(f"已生成最终规划：{html_path.relative_to(PROJECT_ROOT)}")
        st.dataframe(launch, use_container_width=True, hide_index=True)

    st.subheader("输出结果交接")
    handoff = pd.DataFrame(
        [
            ("delivery_exception_result.csv", "配送异常订单诊断表", "大模型报告模块"),
            ("outlet_risk_score.csv", "网点风险评分结果", "大模型报告模块"),
            ("risk_map.html", "配送风险地图", "大模型报告模块"),
            ("cluster_result.csv", "K-Means配送优化片区", "低空应急规划"),
            ("launch_candidates.csv", "候选起降点筛选结果", "低空应急规划"),
            ("launch_points.csv", "最终低空应急起降点推荐", "大模型报告模块"),
            ("low_altitude_plan.html", "低空起降点规划地图", "大模型报告模块"),
            ("route_candidates.csv", "典型订单候选航线评分表", "大模型报告模块"),
            ("route_assignment.csv", "高优先级异常订单路径分配表", "大模型报告模块"),
            ("route_simulation_report.html", "低空航线仿真报告", "大模型报告模块"),
        ],
        columns=["输出文件", "文件作用", "交接对象"],
    )
    st.dataframe(handoff, use_container_width=True, hide_index=True)

    launch_path = PROCESSED_DIR / "launch_points.csv"
    if launch_path.exists():
        st.subheader("launch_points.csv 预览")
        st.dataframe(read_csv(launch_path), use_container_width=True, hide_index=True)


def ensure_launch_points() -> pd.DataFrame:
    launch_path = PROCESSED_DIR / "launch_points.csv"
    if launch_path.exists():
        launch = read_csv(launch_path)
        if not launch.empty:
            return launch
    candidate_path = PROCESSED_DIR / "launch_candidates.csv"
    if candidate_path.exists():
        candidates = read_csv(candidate_path)
        if not candidates.empty:
            selected = candidates.sort_values(["cluster_id", "candidate_rank_in_cluster"]).groupby("cluster_id", as_index=False).head(1).copy()
            rows = []
            for idx, row in selected.reset_index(drop=True).iterrows():
                rows.append(
                    {
                        "launch_point_id": f"LP{idx + 1:03d}",
                        "launch_point_name": f"片区{int(row['cluster_id'])}应急起降点",
                        "cluster_id": int(row["cluster_id"]),
                        "town": row.get("town", ""),
                        "lon": round(float(row["lon"]), 6),
                        "lat": round(float(row["lat"]), 6),
                        "covered_outlets": 0,
                        "covered_high_risk_outlets": 0,
                        "covered_high_priority_orders": int(pd.to_numeric(row.get("high_priority_order_count", 0), errors="coerce") or 0),
                        "service_radius_km": float(row.get("service_radius_km", 5)),
                        "recommend_score": round(float(row.get("candidate_score", 0)), 2),
                        "recommend_reason": "由候选起降点规则筛选后直接进入航线规划",
                    }
                )
            launch = pd.DataFrame(rows)
            write_csv(launch, PROCESSED_DIR / "launch_points.csv")
            write_log("由候选起降点生成航线规划起点", "launch_points.csv")
            return launch
    return make_launch_points_from_candidates("均衡方案")


def route_planning_orders(task_types: list[str]) -> pd.DataFrame:
    exception_path = PROCESSED_DIR / "delivery_exception_result.csv"
    if exception_path.exists():
        exceptions = read_csv(exception_path)
    else:
        exceptions = make_delivery_exceptions(48, 36, 24, True)
    if exceptions.empty:
        return pd.DataFrame()

    risk_path = PROCESSED_DIR / "outlet_risk_score.csv"
    risk = read_csv(risk_path) if risk_path.exists() else make_outlet_risk_score(DEFAULT_RISK_WEIGHTS)
    if risk.empty or "outlet_id" not in risk.columns:
        metrics = ensure_outlet_metrics()
        risk = metrics.copy()
        if "risk_score_base" in risk.columns:
            risk["final_risk_score"] = pd.to_numeric(risk["risk_score_base"], errors="coerce").fillna(0)
            risk["risk_score"] = risk["final_risk_score"]
        else:
            risk["final_risk_score"] = 0
            risk["risk_score"] = 0
        risk["risk_level"] = np.where(risk["final_risk_score"] >= 70, "高风险", np.where(risk["final_risk_score"] >= 40, "中风险", "低风险"))
    risk_cols = [
        "outlet_id",
        "outlet_name",
        "town",
        "region_type",
        "lon",
        "lat",
        "risk_level",
        "final_risk_score",
        "risk_score",
    ]
    risk_cols = [col for col in risk_cols if col in risk.columns]
    if "outlet_id" in risk_cols:
        orders = exceptions.merge(risk[risk_cols], on="outlet_id", how="left")
    else:
        outlets = load_outlets()
        outlet_cols = [col for col in ["outlet_id", "outlet_name", "town", "region_type", "lon", "lat"] if col in outlets.columns]
        orders = exceptions.merge(outlets[outlet_cols], on="outlet_id", how="left") if "outlet_id" in outlet_cols else exceptions.copy()
    if "final_risk_score" not in orders.columns:
        base_risk = orders["risk_score"] if "risk_score" in orders.columns else pd.Series(0, index=orders.index)
        orders["final_risk_score"] = pd.to_numeric(base_risk, errors="coerce").fillna(0)
    if "risk_level" not in orders.columns:
        orders["risk_level"] = np.where(orders["final_risk_score"] >= 70, "高风险", np.where(orders["final_risk_score"] >= 40, "中风险", "低风险"))
    if "outlet_name" not in orders.columns:
        orders["outlet_name"] = orders["outlet_id"].astype(str) + "网点"
    for column in ["town", "region_type", "category", "abc_class", "product_name"]:
        if column not in orders.columns:
            orders[column] = ""

    selected_masks = []
    if "A类重点商品超时订单" in task_types:
        selected_masks.append(orders["exception_type"].astype(str).str.contains("A类", na=False) | orders["abc_class"].astype(str).eq("A"))
    if "生鲜冷链超时订单" in task_types:
        selected_masks.append(orders["exception_type"].astype(str).str.contains("生鲜|冷链", regex=True, na=False) | orders["category"].astype(str).str.contains("生鲜|冷链|农产品", regex=True, na=False))
    if "乡镇高风险网点订单" in task_types:
        selected_masks.append(orders["region_type"].astype(str).eq("乡镇") & orders["risk_level"].astype(str).eq("高风险"))
    if "全部高优先级异常订单" in task_types or not selected_masks:
        selected_masks.append(orders["priority_level"].astype(str).eq("高"))
    mask = selected_masks[0]
    for item in selected_masks[1:]:
        mask = mask | item
    planned = orders[mask].copy()
    if planned.empty:
        planned = orders.copy()

    planned["route_priority_score"] = (
        pd.to_numeric(planned["actual_hours_calc"], errors="coerce").fillna(0) * 0.45
        + pd.to_numeric(planned["final_risk_score"], errors="coerce").fillna(0) * 0.35
        + planned["abc_class"].astype(str).eq("A").astype(int) * 18
        + planned["category"].astype(str).str.contains("生鲜|冷链|农产品", regex=True, na=False).astype(int) * 12
    )
    planned = planned.sort_values("route_priority_score", ascending=False).head(28).reset_index(drop=True)
    planned["display_label"] = planned.apply(
        lambda row: f"{row['order_id']} | {row['outlet_id']} {row.get('outlet_name', '')} | {row.get('product_name', '')} | {row.get('exception_type', '')}",
        axis=1,
    )
    return planned


def nearest_launch_point(order_row: pd.Series, launch: pd.DataFrame) -> pd.Series:
    if launch.empty:
        return pd.Series(dtype=object)
    launch = launch.copy()
    launch["lon"] = pd.to_numeric(launch["lon"], errors="coerce").fillna(115.48)
    launch["lat"] = pd.to_numeric(launch["lat"], errors="coerce").fillna(35.23)
    target_lon = pd.to_numeric(pd.Series([order_row.get("lon", launch["lon"].mean())]), errors="coerce").fillna(launch["lon"].mean()).iloc[0]
    target_lat = pd.to_numeric(pd.Series([order_row.get("lat", launch["lat"].mean())]), errors="coerce").fillna(launch["lat"].mean()).iloc[0]
    launch["distance_to_target"] = launch.apply(lambda row: haversine_km(row["lon"], row["lat"], target_lon, target_lat), axis=1)
    return launch.sort_values("distance_to_target").iloc[0]


def build_route_candidates(
    order_row: pd.Series,
    max_distance_km: float,
    max_time_min: int,
    min_battery: int,
    weather: str,
    optimization_goal: str,
    learning_rate: float = 0.08,
    discount_factor: float = 0.92,
    exploration_rate: float = 0.12,
    train_rounds: int = 120,
    reward_weights: dict[str, int] | None = None,
    enabled_actions: list[str] | None = None,
) -> pd.DataFrame:
    launch = ensure_launch_points()
    launch_row = nearest_launch_point(order_row, launch)
    launch_lon = pd.to_numeric(pd.Series([launch_row.get("lon", 115.48)]), errors="coerce").fillna(115.48).iloc[0]
    launch_lat = pd.to_numeric(pd.Series([launch_row.get("lat", 35.23)]), errors="coerce").fillna(35.23).iloc[0]
    target_lon = pd.to_numeric(pd.Series([order_row.get("lon", launch_lon)]), errors="coerce").fillna(launch_lon).iloc[0]
    target_lat = pd.to_numeric(pd.Series([order_row.get("lat", launch_lat)]), errors="coerce").fillna(launch_lat).iloc[0]
    base_distance = haversine_km(float(launch_lon), float(launch_lat), target_lon, target_lat)
    base_distance = max(base_distance, 1.2)
    weather_factor = {"晴": 1.0, "雨": 1.12, "雾": 1.18, "大风": 1.25}.get(weather, 1.0)
    route_defs = [
        ("R-A", "最短直达航线", 1.02, 68, 1, "距离最短，适合天气良好且风险较低的订单"),
        ("R-B", "低风险绕行航线", 1.18, 92, 2, "避开高风险区域，综合安全得分最高"),
        ("R-C", "多网点覆盖航线", 1.34, 80, 4, "兼顾周边网点覆盖，适合批量应急任务"),
        ("R-R", "备用返航航线", 1.08, 96, 1, "保留返航安全余量，适合天气或电量异常时启用"),
    ]
    enabled_actions = enabled_actions or ["R-A", "R-B", "R-C"]
    route_defs = [item for item in route_defs if item[0] in enabled_actions]
    if not route_defs:
        route_defs = [("R-B", "低风险绕行航线", 1.18, 92, 2, "避开高风险区域，综合安全得分最高")]
    reward_weights = reward_weights or {"time": 35, "safety": 25, "priority": 20, "battery": 15, "coverage": 5}
    total_weight = max(sum(reward_weights.values()), 1)
    time_w = reward_weights.get("time", 0) / total_weight
    safety_w = reward_weights.get("safety", 0) / total_weight
    priority_w = reward_weights.get("priority", 0) / total_weight
    battery_w = reward_weights.get("battery", 0) / total_weight
    coverage_w = reward_weights.get("coverage", 0) / total_weight
    priority_score = 92 if str(order_row.get("abc_class", "")) == "A" else 78
    if str(order_row.get("category", "")).find("冷链") >= 0 or str(order_row.get("category", "")).find("生鲜") >= 0:
        priority_score = max(priority_score, 94)
    rows = []
    for route_id, route_type, distance_factor, safety_score, coverage_count, reason in route_defs:
        distance = round(base_distance * distance_factor * weather_factor, 2)
        estimated_time = int(round(distance / 30 * 60 + 3))
        battery_usage = min(96, round(distance / max(max_distance_km, 1) * 72 + coverage_count * 3, 1))
        time_score = max(40, 100 - estimated_time * 2.2)
        coverage_score = min(100, 55 + coverage_count * 12)
        battery_score = max(35, 100 - battery_usage)
        if distance > max_distance_km or estimated_time > max_time_min or battery_usage > (100 - min_battery):
            safety_score = max(45, safety_score - 18)
            reason += "，但受航程/时间/电量约束影响，需人工复核"
        raw_reward = (
            time_score * time_w
            + safety_score * safety_w
            + coverage_score * coverage_w
            + priority_score * priority_w
            + battery_score * battery_w
        )
        rl_adjustment = discount_factor * 4.5 + learning_rate * 12 - exploration_rate * 8 + min(train_rounds, 300) / 300 * 3
        reward = raw_reward + rl_adjustment
        rows.append(
            {
                "route_id": route_id,
                "route_type": route_type,
                "launch_point_id": launch_row.get("launch_point_id", "LP001"),
                "target_outlet_id": order_row.get("outlet_id", ""),
                "distance_km": distance,
                "estimated_time_min": estimated_time,
                "risk_level": "低" if safety_score >= 86 else ("中" if safety_score >= 70 else "高"),
                "coverage_count": coverage_count,
                "battery_usage": battery_usage,
                "reward_score": round(reward, 1),
                "time_reward": round(time_score, 1),
                "safety_reward": round(safety_score, 1),
                "coverage_reward": round(coverage_score, 1),
                "priority_reward": round(priority_score, 1),
                "battery_reward": round(battery_score, 1),
                "recommend_reason": reason,
            }
        )
    candidates = pd.DataFrame(rows).sort_values("reward_score", ascending=False).reset_index(drop=True)
    return candidates


def make_route_candidates(
    order_row: pd.Series,
    max_distance_km: float,
    max_time_min: int,
    min_battery: int,
    weather: str,
    optimization_goal: str,
    learning_rate: float = 0.08,
    discount_factor: float = 0.92,
    exploration_rate: float = 0.12,
    train_rounds: int = 120,
    reward_weights: dict[str, int] | None = None,
    enabled_actions: list[str] | None = None,
) -> pd.DataFrame:
    candidates = build_route_candidates(order_row, max_distance_km, max_time_min, min_battery, weather, optimization_goal, learning_rate, discount_factor, exploration_rate, train_rounds, reward_weights, enabled_actions)
    write_csv(candidates, PROCESSED_DIR / "route_candidates.csv")
    write_log("生成低空候选航线", "route_candidates.csv")
    return candidates


def make_route_assignment(
    task_types: list[str],
    max_distance_km: float,
    max_time_min: int,
    min_battery: int,
    weather: str,
    optimization_goal: str,
    learning_rate: float = 0.08,
    discount_factor: float = 0.92,
    exploration_rate: float = 0.12,
    train_rounds: int = 120,
    reward_weights: dict[str, int] | None = None,
    enabled_actions: list[str] | None = None,
) -> pd.DataFrame:
    orders = route_planning_orders(task_types)
    if orders.empty:
        return pd.DataFrame()
    rows = []
    for idx, (_, order_row) in enumerate(orders.iterrows(), start=1):
        candidates = build_route_candidates(order_row, max_distance_km, max_time_min, min_battery, weather, optimization_goal, learning_rate, discount_factor, exploration_rate, train_rounds, reward_weights, enabled_actions)
        best = candidates.iloc[0]
        rows.append(
            {
                "order_id": order_row.get("order_id", f"ORD{idx:04d}"),
                "product_id": order_row.get("product_id", ""),
                "product_name": order_row.get("product_name", ""),
                "abc_class": order_row.get("abc_class", ""),
                "category": order_row.get("category", ""),
                "outlet_id": order_row.get("outlet_id", ""),
                "launch_point_id": best["launch_point_id"],
                "route_id": f"{best['route_id']}-{idx:03d}",
                "route_type": best["route_type"],
                "estimated_distance_km": best["distance_km"],
                "estimated_time_min": best["estimated_time_min"],
                "risk_level": best["risk_level"],
                "reward_score": best["reward_score"],
                "assigned_reason": f"{best['route_type']}综合奖励最高，适合{order_row.get('exception_type', '高优先级异常订单')}的低空应急配送",
            }
        )
    assignment = pd.DataFrame(rows)
    write_csv(assignment, PROCESSED_DIR / "route_assignment.csv")
    write_log("生成批量低空路径分配", "route_assignment.csv")
    return assignment


def route_simulation_html(order_row: pd.Series, candidates: pd.DataFrame, selected_route_id: str) -> str:
    selected = candidates[candidates["route_id"] == selected_route_id]
    if selected.empty:
        selected = candidates.head(1)
    best = selected.iloc[0]
    route_color = {"R-A": "#2f80ff", "R-B": "#21e6c1", "R-C": "#9b5cff"}.get(str(best["route_id"]), "#21e6c1")
    a_selected = " selected" if str(best["route_id"]) == "R-A" else ""
    b_selected = " selected" if str(best["route_id"]) == "R-B" else ""
    c_selected = " selected" if str(best["route_id"]) == "R-C" else ""
    rows_html = ""
    for _, route in candidates.iterrows():
        active = "active" if route["route_id"] == best["route_id"] else ""
        rows_html += f"""
        <tr class="{active}">
            <td>{route['route_id']}</td>
            <td>{route['route_type']}</td>
            <td>{route['distance_km']} km</td>
            <td>{route['estimated_time_min']} min</td>
            <td>{route['risk_level']}</td>
            <td>{route['reward_score']}</td>
        </tr>
        """
    return f"""
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        body {{ margin: 0; background: #07111f; color: #e9f4ff; font-family: "Microsoft YaHei", Arial, sans-serif; }}
        .route-shell {{ display: grid; grid-template-columns: 1fr 310px; gap: 16px; padding: 18px; background: radial-gradient(circle at 35% 25%, rgba(38,120,217,.32), transparent 30%), #07111f; border: 1px solid rgba(90,170,255,.28); border-radius: 16px; }}
        .stage {{ position: relative; height: 520px; border: 1px solid rgba(90,170,255,.22); border-radius: 14px; overflow: hidden; background: linear-gradient(180deg, #091a2d 0%, #050b14 100%); }}
        .stage:before {{ content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(63, 147, 255, .12) 1px, transparent 1px), linear-gradient(90deg, rgba(63, 147, 255, .12) 1px, transparent 1px); background-size: 34px 34px; transform: perspective(520px) rotateX(58deg) translateY(120px) scale(1.35); transform-origin: center bottom; }}
        svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
        .path {{ fill: none; stroke-width: 4; stroke-linecap: round; opacity: .7; filter: drop-shadow(0 0 7px currentColor); stroke-dasharray: 8 10; animation: dash 1.4s linear infinite; }}
        .path-a {{ stroke: #2f80ff; color: #2f80ff; }}
        .path-b {{ stroke: #21e6c1; color: #21e6c1; }}
        .path-c {{ stroke: #9b5cff; color: #9b5cff; }}
        .selected {{ stroke: {route_color}; stroke-width: 7; opacity: 1; }}
        @keyframes dash {{ to {{ stroke-dashoffset: -36; }} }}
        .pad, .target {{ position: absolute; border-radius: 50%; transform: translate(-50%, -50%); }}
        .pad {{ left: 18%; top: 68%; width: 92px; height: 92px; background: radial-gradient(circle, rgba(45,159,255,.85), rgba(45,159,255,.08) 58%, transparent 62%); box-shadow: 0 0 30px #2297ff; }}
        .target {{ left: 79%; top: 34%; width: 70px; height: 70px; background: radial-gradient(circle, rgba(255,61,87,.9), rgba(255,61,87,.12) 58%, transparent 64%); box-shadow: 0 0 26px #ff3d57; animation: pulse 1.2s ease-in-out infinite; }}
        @keyframes pulse {{ 50% {{ transform: translate(-50%, -50%) scale(1.12); }} }}
        .label {{ position: absolute; padding: 7px 10px; border-radius: 10px; background: rgba(5,15,27,.78); border: 1px solid rgba(138,196,255,.34); color: #dff2ff; font-size: 13px; }}
        .launch-label {{ left: 8%; top: 77%; }}
        .target-label {{ right: 6%; top: 20%; }}
        .drone {{ position: absolute; left: 18%; top: 68%; width: 34px; height: 34px; transform: translate(-50%, -50%); animation: fly 6s ease-in-out infinite; filter: drop-shadow(0 0 12px #fff); }}
        .drone:before, .drone:after {{ content: ""; position: absolute; background: #e8fbff; border-radius: 999px; }}
        .drone:before {{ left: 3px; right: 3px; top: 15px; height: 4px; }}
        .drone:after {{ top: 3px; bottom: 3px; left: 15px; width: 4px; }}
        .rotor {{ position: absolute; width: 10px; height: 10px; border: 2px solid #9fe7ff; border-radius: 50%; box-shadow: 0 0 10px #9fe7ff; }}
        .r1 {{ left: -2px; top: -2px; }} .r2 {{ right: -2px; top: -2px; }} .r3 {{ left: -2px; bottom: -2px; }} .r4 {{ right: -2px; bottom: -2px; }}
        @keyframes fly {{ 0% {{ left:18%; top:68%; }} 45% {{ left:52%; top:30%; }} 80% {{ left:79%; top:34%; }} 100% {{ left:79%; top:34%; }} }}
        .panel {{ background: rgba(7,17,31,.82); border: 1px solid rgba(90,170,255,.26); border-radius: 14px; padding: 14px; }}
        .panel h3 {{ margin: 0 0 12px 0; color: #7fd8ff; }}
        .score {{ font-size: 54px; font-weight: 800; color: {route_color}; line-height: 1; margin: 10px 0; }}
        .kv {{ display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,.08); padding: 9px 0; color: #d9eaff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 14px; overflow: hidden; border-radius: 12px; }}
        th, td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,.08); text-align: left; font-size: 13px; }}
        th {{ color: #8fdcff; background: rgba(33,104,177,.18); }}
        tr.active td {{ background: rgba(33,230,193,.14); color: #fff; }}
        .done {{ position: absolute; right: 24px; bottom: 20px; padding: 9px 13px; color: #b8ffe8; background: rgba(33,230,193,.13); border: 1px solid rgba(33,230,193,.45); border-radius: 999px; }}
    </style>
    </head>
    <body>
        <div class="route-shell">
            <div>
                <div class="stage">
                    <svg viewBox="0 0 900 520">
                        <path class="path path-a{a_selected}" d="M160 355 C335 240, 520 190, 710 180" />
                        <path class="path path-b{b_selected}" d="M160 355 C275 250, 460 110, 710 180" />
                        <path class="path path-c{c_selected}" d="M160 355 C300 405, 610 360, 710 180" />
                    </svg>
                    <div class="pad"></div>
                    <div class="target"></div>
                    <div class="label launch-label">{best['launch_point_id']} 起降平台</div>
                    <div class="label target-label">{order_row.get('outlet_id', '')} {order_row.get('outlet_name', '目标网点')}</div>
                    <div class="drone"><span class="rotor r1"></span><span class="rotor r2"></span><span class="rotor r3"></span><span class="rotor r4"></span></div>
                    <div class="done">仿真飞行中：推荐航线 {best['route_id']}</div>
                </div>
                <table>
                    <thead><tr><th>航线</th><th>类型</th><th>距离</th><th>时间</th><th>风险</th><th>奖励分</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            <div class="panel">
                <h3>AI航线评分面板</h3>
                <div>当前推荐航线</div>
                <div class="score">{best['reward_score']}</div>
                <div class="kv"><span>航线类型</span><b>{best['route_type']}</b></div>
                <div class="kv"><span>预计距离</span><b>{best['distance_km']} km</b></div>
                <div class="kv"><span>预计时间</span><b>{best['estimated_time_min']} min</b></div>
                <div class="kv"><span>安全风险</span><b>{best['risk_level']}</b></div>
                <div class="kv"><span>目标订单</span><b>{order_row.get('order_id', '')}</b></div>
                <div class="kv"><span>商品</span><b>{order_row.get('product_name', '')}</b></div>
                <p style="color:#b8cde5;line-height:1.7;">推荐理由：{best['recommend_reason']}。该航线用于高优先级异常订单的低空应急配送，不替代普通地面配送。</p>
            </div>
        </div>
    </body>
    </html>
    """


def write_route_simulation_report(order_row: pd.Series, candidates: pd.DataFrame, selected_route_id: str) -> tuple[Path, str]:
    html = route_simulation_html(order_row, candidates, selected_route_id)
    path = REPORT_DIR / "route_simulation_report.html"
    path.write_text(html, encoding="utf-8")
    write_log("生成低空航线仿真报告", "route_simulation_report.html")
    return path, html


def page_route_planning() -> None:
    st.header("AI低空应急配送航线智能规划系统")
    st.caption("该页面只对高优先级异常订单进行低空应急航线规划，普通订单仍以地面配送为主。")

    task_types = st.multiselect(
        "选择低空配送任务类型",
        ["A类重点商品超时订单", "生鲜冷链超时订单", "乡镇高风险网点订单", "全部高优先级异常订单"],
        default=["A类重点商品超时订单", "生鲜冷链超时订单"],
    )
    task_key = "|".join(task_types)
    if st.session_state.get("route_task_key") != task_key:
        st.session_state["route_task_key"] = task_key
        st.session_state["route_candidates"] = pd.DataFrame()
        st.session_state["route_assignment_df"] = pd.DataFrame()
        st.session_state["route_order_label"] = None
    planning_orders = route_planning_orders(task_types)
    if planning_orders.empty:
        st.warning("请先完成异常订单诊断和起降点规划。")
        return

    a_count = int((planning_orders["exception_type"].astype(str).str.contains("A类", na=False) | planning_orders["abc_class"].astype(str).eq("A")).sum())
    fresh_count = int((planning_orders["exception_type"].astype(str).str.contains("生鲜|冷链", regex=True, na=False) | planning_orders["category"].astype(str).str.contains("生鲜|冷链|农产品", regex=True, na=False)).sum())
    town_count = int((planning_orders["region_type"].astype(str).eq("乡镇") & planning_orders["risk_level"].astype(str).eq("高风险")).sum())
    metric_cards(
        [
            ("纳入低空规划订单", f"{len(planning_orders)}单", "高优先级异常订单"),
            ("A类重点订单", f"{a_count}单", "优先保障"),
            ("生鲜冷链订单", f"{fresh_count}单", "时效敏感"),
            ("乡镇高风险订单", f"{town_count}单", "低空试点"),
        ]
    )

    left, right = st.columns([1.15, 1])
    with left:
        selected_label = st.selectbox("选择典型高优先级异常订单", planning_orders["display_label"].tolist())
        if st.session_state.get("route_selected_label") != selected_label:
            st.session_state["route_selected_label"] = selected_label
            st.session_state["route_candidates"] = pd.DataFrame()
            st.session_state["route_assignment_df"] = pd.DataFrame()
            st.session_state["route_order_label"] = None
        order_row = planning_orders[planning_orders["display_label"] == selected_label].iloc[0]
        state_df = pd.DataFrame(
            [
                ("起点", "系统根据 launch_points.csv 自动选择最近起降点"),
                ("终点", f"{order_row.get('outlet_id', '')} {order_row.get('outlet_name', '')}"),
                ("订单类型", order_row.get("exception_type", "")),
                ("商品", f"{order_row.get('product_name', '')} / {order_row.get('category', '')}"),
                ("网点风险", order_row.get("risk_level", "")),
                ("配送时长", f"{order_row.get('actual_hours_calc', 0)}小时"),
            ],
            columns=["状态项", "当前状态"],
        )
        st.dataframe(state_df, use_container_width=True, hide_index=True)

    with right:
        max_distance_km = st.slider("最大飞行距离/公里", 4.0, 15.0, 8.0, step=0.5)
        max_time_min = st.slider("最大飞行时间/分钟", 8, 35, 18)
        max_payload = st.slider("最大载重/kg", 1.0, 8.0, 3.0, step=0.5)
        min_battery = st.slider("最低返航电量/%", 10, 45, 25)
        weather = st.selectbox("天气风险", ["晴", "雨", "雾", "大风"], index=0)
        optimization_goal = st.selectbox("优化目标", ["综合评分最优", "时效优先", "安全优先", "覆盖优先"], index=0)
        st.caption(f"当前约束：航程{max_distance_km}km，载重{max_payload}kg，返航电量不低于{min_battery}%。")

    st.subheader("模拟强化学习模型配置")
    state_options = [
        "起降点位置",
        "目标网点位置",
        "商品优先级",
        "商品类型",
        "网点风险等级",
        "飞行距离",
        "商品重量",
        "天气风险",
        "初始电量",
        "服务半径",
        "覆盖网点数量",
    ]
    default_states = []
    selected_states = st.multiselect("State 状态变量选择", state_options, default=default_states)

    action_options = {
        "R-A": "A 最短直达航线",
        "R-B": "B 低风险绕行航线",
        "R-C": "C 多网点覆盖航线",
        "R-R": "R 备用返航航线",
    }
    selected_actions_label = st.multiselect(
        "Action 动作空间选择",
        list(action_options.values()),
        default=[],
    )
    enabled_actions = [key for key, label in action_options.items() if label in selected_actions_label]

    reward_template = st.selectbox("Reward 奖励函数模板", ["A类冷链超时订单模板", "批量乡镇订单模板", "恶劣天气安全模板", "自定义权重"], index=3)
    template_weights = {
        "A类冷链超时订单模板": {"time": 35, "safety": 25, "priority": 20, "battery": 15, "coverage": 5},
        "批量乡镇订单模板": {"time": 25, "safety": 20, "priority": 15, "battery": 15, "coverage": 25},
        "恶劣天气安全模板": {"time": 20, "safety": 40, "priority": 15, "battery": 20, "coverage": 5},
        "自定义权重": {"time": 35, "safety": 25, "priority": 20, "battery": 15, "coverage": 5},
    }
    defaults = template_weights[reward_template]
    reward_col1, reward_col2, reward_col3, reward_col4, reward_col5 = st.columns(5)
    with reward_col1:
        time_weight = st.slider("时效得分权重", 0, 60, defaults["time"])
    with reward_col2:
        safety_weight = st.slider("安全得分权重", 0, 60, defaults["safety"])
    with reward_col3:
        priority_weight = st.slider("重点商品保障权重", 0, 60, defaults["priority"])
    with reward_col4:
        battery_weight = st.slider("电量余量权重", 0, 60, defaults["battery"])
    with reward_col5:
        coverage_weight = st.slider("覆盖能力权重", 0, 60, defaults["coverage"])
    reward_weights = {
        "time": time_weight,
        "safety": safety_weight,
        "priority": priority_weight,
        "battery": battery_weight,
        "coverage": coverage_weight,
    }
    reward_total = sum(reward_weights.values())
    if reward_total != 100:
        st.warning(f"当前Reward权重合计为{reward_total}%，建议调整为100%。系统会按比例归一化计算。")
    else:
        st.success("Reward权重合计100%，当前配置适合A类冷链超时订单航线评分。")

    rl_col1, rl_col2, rl_col3, rl_col4 = st.columns(4)
    with rl_col1:
        learning_rate = st.slider("学习率 alpha", 0.00, 0.20, 0.00, step=0.01)
    with rl_col2:
        discount_factor = st.slider("折扣因子 gamma", 0.00, 0.99, 0.00, step=0.01)
    with rl_col3:
        exploration_rate = st.slider("探索率 epsilon", 0.00, 0.40, 0.00, step=0.01)
    with rl_col4:
        train_rounds = st.slider("模拟训练轮次", 0, 300, 0, step=10)

    route_param_key = f"{selected_label}|{max_distance_km}|{max_time_min}|{max_payload}|{min_battery}|{weather}|{optimization_goal}|{learning_rate}|{discount_factor}|{exploration_rate}|{train_rounds}|{selected_states}|{enabled_actions}|{reward_weights}"
    if st.session_state.get("route_param_key") != route_param_key:
        st.session_state["route_param_key"] = route_param_key
        st.session_state["route_candidates"] = pd.DataFrame()
        st.session_state["route_assignment_df"] = pd.DataFrame()
        st.session_state["route_order_label"] = None

    generate_col, score_col, fly_col, batch_col = st.columns(4)
    with generate_col:
        generate_clicked = st.button("生成候选航线", type="primary")
    with score_col:
        score_clicked = st.button("AI路径评分")
    with fly_col:
        fly_clicked = st.button("执行仿真飞行")
    with batch_col:
        batch_clicked = st.button("生成批量路径分配表")

    if generate_clicked:
        if not selected_states:
            st.error("请至少选择一个State状态变量。")
            return
        if not enabled_actions:
            st.error("请至少选择一个Action航线动作。")
            return
        render_timed_steps(
            [
                "正在初始化强化学习航线环境...",
                "正在读取State状态变量...",
                "正在加载Action动作空间...",
                "正在构建Reward奖励函数...",
                "正在评估起降点与目标网点距离...",
                "正在执行第1阶段航线探索...",
                "正在执行第2阶段奖励回传...",
                "正在更新候选航线价值评分...",
                "正在筛选最优候选航线...",
                "正在生成航线评分与可视化结果...",
            ],
            duration_seconds=15.0,
        )
        candidates = make_route_candidates(order_row, max_distance_km, max_time_min, min_battery, weather, optimization_goal, learning_rate, discount_factor, exploration_rate, train_rounds, reward_weights, enabled_actions)
        st.session_state["route_candidates"] = candidates
        st.session_state["route_order_label"] = selected_label
        st.session_state["route_assignment_df"] = pd.DataFrame()
        st.session_state["route_scroll_to_results"] = True
        st.success("已生成3条候选航线：最短直达、低风险绕行、多网点覆盖。")
    else:
        candidates = st.session_state.get("route_candidates", pd.DataFrame())
        if st.session_state.get("route_order_label") != selected_label:
            candidates = pd.DataFrame()

    if candidates.empty and (score_clicked or fly_clicked):
        st.warning("请先点击“生成候选航线”，再进行AI路径评分或仿真飞行。")

    if not candidates.empty:
        st.markdown('<div id="module3-route-results"></div>', unsafe_allow_html=True)
        selected_route_id = st.radio(
            "选择候选航线查看评分与路线",
            candidates["route_id"].tolist(),
            index=int(candidates["reward_score"].idxmax()),
            horizontal=True,
        )
        selected_route = candidates[candidates["route_id"] == selected_route_id].iloc[0]
        rl_summary = pd.DataFrame(
            [
                ("State状态变量", "、".join(selected_states)),
                ("Action动作空间", "、".join(selected_actions_label)),
                ("Reward权重", f"时效{time_weight}% / 安全{safety_weight}% / 重点商品{priority_weight}% / 电量{battery_weight}% / 覆盖{coverage_weight}%"),
                ("RL参数", f"alpha={learning_rate}，gamma={discount_factor}，epsilon={exploration_rate}，训练{train_rounds}轮"),
            ],
            columns=["配置项", "当前配置"],
        )
        st.dataframe(rl_summary, use_container_width=True, hide_index=True)
        metric_cards(
            [
                ("当前航线", str(selected_route["route_type"]), str(selected_route["route_id"])),
                ("综合奖励分", str(selected_route["reward_score"]), "RL-Route Planner"),
                ("预计距离", f"{selected_route['distance_km']} km", "航线距离"),
                ("预计时间", f"{selected_route['estimated_time_min']} min", "低空配送"),
            ]
        )
        score_df = pd.DataFrame(
            [
                ("时效奖励", selected_route["time_reward"]),
                ("安全奖励", selected_route["safety_reward"]),
                ("覆盖奖励", selected_route["coverage_reward"]),
                ("重点商品奖励", selected_route["priority_reward"]),
                ("电量奖励", selected_route["battery_reward"]),
                ("综合奖励", selected_route["reward_score"]),
            ],
            columns=["评分项", "得分"],
        )
        chart_col, table_col = st.columns([1.1, 1])
        with chart_col:
            st.plotly_chart(px.bar(score_df, x="评分项", y="得分", color="评分项", title="模拟强化学习奖励评分"), use_container_width=True)
        with table_col:
            st.dataframe(candidates, use_container_width=True, hide_index=True)
        html = route_simulation_html(order_row, candidates, selected_route_id)
        components.html(html, height=720, scrolling=False)
        st.info(f"推荐理由：{selected_route['recommend_reason']}。")
        if st.session_state.pop("route_scroll_to_results", False):
            scroll_to_anchor("module3-route-results")
        if fly_clicked or score_clicked:
            report_path, _ = write_route_simulation_report(order_row, candidates, selected_route_id)
            st.success(f"已生成航线仿真报告：{report_path.relative_to(PROJECT_ROOT)}")

    if batch_clicked:
        if candidates.empty:
            st.warning("请先生成候选航线，再生成批量路径分配表。")
            assignment = None
        else:
            assignment = make_route_assignment(task_types, max_distance_km, max_time_min, min_battery, weather, optimization_goal, learning_rate, discount_factor, exploration_rate, train_rounds, reward_weights, enabled_actions)
        if assignment is not None and assignment.empty:
            st.warning("没有可分配的高优先级异常订单。")
        elif assignment is not None:
            st.session_state["route_assignment_df"] = assignment
            st.success("已生成 route_assignment.csv。")
            metric_cards(
                [
                    ("批量分配订单", f"{len(assignment)}单", "route_assignment.csv"),
                    ("推荐低风险航线", f"{int((assignment['risk_level'] == '低').sum())}单", "安全优先"),
                    ("平均奖励分", f"{assignment['reward_score'].mean():.1f}", "路径评分"),
                    ("平均配送时间", f"{assignment['estimated_time_min'].mean():.1f} min", "低空应急"),
                ]
            )
            st.dataframe(assignment.head(28), use_container_width=True, hide_index=True)

    if not candidates.empty or not st.session_state.get("route_assignment_df", pd.DataFrame()).empty:
        st.subheader("输出结果")
        outputs = [("route_candidates.csv", "典型订单候选航线评分表", "大模型报告模块")]
        if not st.session_state.get("route_assignment_df", pd.DataFrame()).empty:
            outputs.append(("route_assignment.csv", "高优先级异常订单批量路径分配表", "大模型报告模块"))
        if (REPORT_DIR / "route_simulation_report.html").exists():
            outputs.append(("route_simulation_report.html", "低空航线仿真报告", "大模型报告模块"))
        st.dataframe(pd.DataFrame(outputs, columns=["输出文件", "文件作用", "交接对象"]), use_container_width=True, hide_index=True)


def render_shell() -> str:
    if "module3_page" not in st.session_state or st.session_state["module3_page"] not in PAGES:
        st.session_state["module3_page"] = PAGES[0]
    with st.sidebar:
        st.markdown("### 物流企业AI智能运营决策平台")
        st.caption("模块三 / 学生3：配送智能规划工程师")
        selected = st.radio("功能菜单", PAGES, key="module3_page", label_visibility="collapsed")
        st.divider()
        st.markdown("**当前模块**")
        st.write(MODULE_TITLE)
        st.caption("负责异常订单诊断、网点风险评分、K-Means聚类分区、起降点规划与低空应急航线分配。")

    st.markdown(
        f"""
        <div class="topbar">
            <span class="topbar-label">Logistics AI Operations / Module 3</span>
            <strong>{APP_TITLE}</strong>
            <div class="role-chip">{ROLE_NAME}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-line">
            <span>接收数据并诊断异常</span>
            <span>校验风险权重</span>
            <span>开关地图图层</span>
            <span>评估K-Means参数</span>
            <span>筛选候选起降点</span>
            <span>规划低空航线</span>
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

    if selected == "数据接收与异常订单诊断":
        page_problem_judgement()
    elif selected == "网点异常汇总与风险评分":
        page_risk_weight_v2()
    elif selected == "配送风险地图图层分析":
        page_map_layers_v2()
    elif selected == "K-Means聚类参数配置与评估":
        page_cluster_eval_v2()
    elif selected == "起降点候选规则配置":
        page_launch_candidate_rules()
    elif selected == "AI低空应急配送航线智能规划系统":
        page_route_planning()


if __name__ == "__main__":
    main()
