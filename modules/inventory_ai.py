from __future__ import annotations

from pathlib import Path

import streamlit as st

from modules.utils import list_files, read_recent_logs, render_workflow_steps, write_log


def run_abc_inventory_model(inventory_file: Path, config: dict) -> dict:
    """Future replacement point for ABC classification and sales forecasting models."""
    raise NotImplementedError("Inventory AI logic will be implemented in the next milestone.")


def render_page(paths: dict[str, Path]) -> None:
    st.header("AI库存分析与ABC决策系统")
    st.markdown(
        '<div class="placeholder-note">本模块当前为占位页面，后续会接入ABC分层、销量预测、滞销预警和补货建议。</div>',
        unsafe_allow_html=True,
    )
    render_workflow_steps("库存AI模块骨架")

    tabs = st.tabs(["导入数据", "参数设置", "运行处理", "查看日志", "展示结果", "导出结果"])

    with tabs[0]:
        st.subheader("导入库存与销量数据")
        st.write("预期读取 `data/raw/inventory.csv`，后续可读取清洗后的销售明细。")
        if st.button("检查库存数据", key="inventory_check_data"):
            write_log("inventory_ai", "检查库存数据")
            files = list_files(paths["raw_data"])
            if files.empty:
                st.warning("未发现模拟数据，请先运行初始化脚本。")
            else:
                st.dataframe(files, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("ABC与预测参数")
        st.slider("A类累计销售额占比", min_value=50, max_value=90, value=70, step=5, key="inventory_a_ratio")
        st.slider("B类累计销售额占比", min_value=75, max_value=98, value=90, step=1, key="inventory_b_ratio")
        st.selectbox("销量预测方式", ["移动平均占位", "线性回归接口", "可替换真实模型"], key="inventory_forecast_method")

    with tabs[2]:
        st.subheader("运行库存分析")
        st.write("下一阶段将在这里生成SKU分层、销量预测和补货建议。")
        if st.button("模拟启动库存AI任务", key="inventory_run"):
            write_log("inventory_ai", "模拟启动库存AI任务")
            st.success("已记录库存AI任务日志。")

    with tabs[3]:
        st.subheader("运行日志")
        logs = read_recent_logs()
        st.code("\n".join(logs) if logs else "暂无日志。", language="text")

    with tabs[4]:
        st.subheader("结果展示占位")
        st.metric("A类SKU数量", "待计算")
        st.metric("预测缺货SKU", "待计算")
        st.metric("建议清仓SKU", "待计算")

    with tabs[5]:
        st.subheader("导出库存决策结果")
        st.write("后续将导出ABC分层表、销量预测表和库存优化建议。")
        if st.button("记录库存导出占位操作", key="inventory_export"):
            write_log("inventory_ai", "记录库存导出占位操作")
            st.success("已记录导出占位操作。")
