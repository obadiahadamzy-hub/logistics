from __future__ import annotations

from pathlib import Path

import streamlit as st

from modules.utils import list_files, read_recent_logs, render_workflow_steps, write_log


def plan_delivery_network(order_file: Path, outlet_file: Path, config: dict) -> dict:
    """Future replacement point for anomaly diagnosis, K-Means clustering, and takeoff-site planning."""
    raise NotImplementedError("Delivery planning logic will be implemented in the next milestone.")


def render_page(paths: dict[str, Path]) -> None:
    st.header("配送异常诊断与低空起降点规划系统")
    st.markdown(
        '<div class="placeholder-note">本模块当前为占位页面，后续会接入异常订单诊断、K-Means网点聚类、地图展示和低空起降点候选评分。</div>',
        unsafe_allow_html=True,
    )
    render_workflow_steps("配送规划模块骨架")

    tabs = st.tabs(["导入数据", "参数设置", "运行处理", "查看日志", "展示结果", "导出结果"])

    with tabs[0]:
        st.subheader("导入订单、网点与候选点数据")
        st.write("预期读取 `orders.csv`、`outlets.csv`、`delivery.csv`、`returns.csv`。")
        if st.button("检查配送规划数据", key="delivery_check_data"):
            write_log("delivery_planning", "检查配送规划数据")
            files = list_files(paths["raw_data"])
            if files.empty:
                st.warning("未发现模拟数据，请先运行初始化脚本。")
            else:
                st.dataframe(files, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("诊断与聚类参数")
        st.slider("异常配送时长阈值（小时）", min_value=12, max_value=120, value=48, step=6, key="delivery_delay_threshold")
        st.slider("K-Means聚类数量", min_value=2, max_value=8, value=4, step=1, key="delivery_cluster_count")
        st.checkbox("启用低空起降点候选评分", value=True, key="delivery_site_score")

    with tabs[2]:
        st.subheader("运行配送规划")
        st.write("下一阶段将在这里输出异常订单列表、网点聚类结果和候选起降点建议。")
        if st.button("模拟启动配送规划任务", key="delivery_run"):
            write_log("delivery_planning", "模拟启动配送规划任务")
            st.success("已记录配送规划任务日志。")

    with tabs[3]:
        st.subheader("运行日志")
        logs = read_recent_logs()
        st.code("\n".join(logs) if logs else "暂无日志。", language="text")

    with tabs[4]:
        st.subheader("地图与聚类结果占位")
        st.metric("异常订单数", "待诊断")
        st.metric("建议聚类网点数", "待计算")
        st.metric("候选起降点", "待评分")

    with tabs[5]:
        st.subheader("导出配送规划结果")
        st.write("后续将导出异常订单表、聚类结果表和低空起降点规划建议。")
        if st.button("记录配送规划导出占位操作", key="delivery_export"):
            write_log("delivery_planning", "记录配送规划导出占位操作")
            st.success("已记录导出占位操作。")
