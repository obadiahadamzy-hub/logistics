from __future__ import annotations

from pathlib import Path

import streamlit as st

from modules.utils import list_files, read_recent_logs, render_workflow_steps, write_log


def process_logistics_data(raw_file: Path, output_dir: Path, config: dict) -> Path:
    """Future replacement point for real data cleaning and BI preprocessing."""
    raise NotImplementedError("Data processing logic will be implemented in the next milestone.")


def render_page(paths: dict[str, Path]) -> None:
    st.header("物流数据处理与BI看板系统")
    st.markdown(
        '<div class="placeholder-note">本模块当前为占位页面，后续会接入订单清洗、指标计算、图表看板和清洗结果导出。</div>',
        unsafe_allow_html=True,
    )
    render_workflow_steps("数据治理模块骨架")

    tabs = st.tabs(["导入数据", "参数设置", "运行处理", "查看日志", "展示结果", "导出结果"])

    with tabs[0]:
        st.subheader("导入原始物流数据")
        st.write("预期读取 `data/raw/orders.csv`、`data/raw/inventory.csv`、`data/raw/outlets.csv`。")
        if st.button("检查原始数据目录", key="data_bi_check_raw"):
            write_log("data_bi", "检查原始数据目录")
            files = list_files(paths["raw_data"])
            if files.empty:
                st.warning("未发现原始数据，请先运行 `python scripts/init_project.py`。")
            else:
                st.dataframe(files, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("数据治理参数")
        st.checkbox("去除重复订单", value=True, key="data_bi_deduplicate")
        st.checkbox("补全缺失乡镇字段", value=True, key="data_bi_fill_town")
        st.slider("异常配送时长阈值（小时）", min_value=12, max_value=120, value=48, step=6, key="data_bi_delay_threshold")

    with tabs[2]:
        st.subheader("运行处理")
        st.write("下一阶段将在这里执行数据清洗，并把结果写入 `data/processed/`。")
        if st.button("模拟启动数据治理任务", key="data_bi_run"):
            write_log("data_bi", "模拟启动数据治理任务")
            st.success("已记录任务启动日志。")

    with tabs[3]:
        st.subheader("运行日志")
        logs = read_recent_logs()
        st.code("\n".join(logs) if logs else "暂无日志。", language="text")

    with tabs[4]:
        st.subheader("BI看板占位")
        st.metric("订单总量", "待生成")
        st.metric("配送异常率", "待计算")
        st.metric("库存周转天数", "待计算")

    with tabs[5]:
        st.subheader("导出结果")
        st.write("后续将支持导出清洗后CSV、BI指标表和图表图片。")
        if st.button("记录导出占位操作", key="data_bi_export"):
            write_log("data_bi", "记录数据治理导出占位操作")
            st.success("已记录导出占位操作。")
