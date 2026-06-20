from __future__ import annotations

from pathlib import Path

import streamlit as st

from modules.utils import list_files, read_recent_logs, render_workflow_steps, write_log


def generate_offline_report(context: dict, template_name: str = "default") -> str:
    """Future replacement point for template reports and DeepSeek/Qwen/Ollama adapters."""
    return "<html><body><h1>经营决策报告占位</h1><p>后续接入模板和大模型接口。</p></body></html>"


def render_page(paths: dict[str, Path]) -> None:
    st.header("大模型经营问答与报告生成系统")
    st.markdown(
        '<div class="placeholder-note">本模块当前为占位页面，第一版将先使用离线模板生成经营报告，并预留DeepSeek、Qwen、Ollama接口。</div>',
        unsafe_allow_html=True,
    )
    render_workflow_steps("大模型报告模块骨架")

    tabs = st.tabs(["导入数据", "参数设置", "运行处理", "查看日志", "展示结果", "导出结果"])

    with tabs[0]:
        st.subheader("导入经营上下文与知识库")
        st.write("预期读取 `data/processed/` 的分析结果和 `data/sample_docs/` 的企业制度文档。")
        if st.button("检查报告输入数据", key="llm_check_data"):
            write_log("llm_report", "检查报告输入数据")
            st.write("原始数据目录")
            st.dataframe(list_files(paths["raw_data"]), use_container_width=True, hide_index=True)
            st.write("知识库文档目录")
            st.dataframe(list_files(paths["sample_docs"]), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("问答与报告参数")
        st.selectbox("运行模式", ["离线模板模式", "DeepSeek API预留", "Qwen API预留", "Ollama本地模型预留"], key="llm_mode")
        st.selectbox("报告类型", ["经营决策简报", "库存优化报告", "配送规划报告", "完整竞赛展示报告"], key="llm_report_type")
        if "llm_report_messages" not in st.session_state:
            st.session_state["llm_report_messages"] = [
                {
                    "role": "assistant",
                    "content": "你好，我是经营报告助手。请输入经营问题，我会基于库存、配送和企业制度知识库生成离线演示回答。",
                }
            ]
        for message in st.session_state["llm_report_messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        question = st.chat_input("输入经营问题，例如：本月哪些SKU需要重点补货？哪些乡镇配送异常较多？", key="llm_report_chat_input")
        if question:
            st.session_state["llm_question"] = question
            st.session_state["llm_report_messages"].append({"role": "user", "content": question})
            st.session_state["llm_report_messages"].append(
                {
                    "role": "assistant",
                    "content": "已接收问题。当前为离线演示模式，后续会结合数据结果和知识库模板生成经营回答与报告。",
                }
            )
            write_log("llm_report", "经营问答对话")
            st.rerun()

    with tabs[2]:
        st.subheader("运行问答与报告生成")
        st.write("下一阶段将在这里调用离线模板，生成HTML经营决策报告。")
        if st.button("模拟生成经营报告", key="llm_run"):
            write_log("llm_report", "模拟生成经营报告")
            st.success("已记录报告生成任务日志。")

    with tabs[3]:
        st.subheader("运行日志")
        logs = read_recent_logs()
        st.code("\n".join(logs) if logs else "暂无日志。", language="text")

    with tabs[4]:
        st.subheader("经营问答结果占位")
        st.write("系统将在这里展示大模型回答、关键指标解释和报告预览。")

    with tabs[5]:
        st.subheader("导出经营决策报告")
        st.write("后续将导出HTML报告，PDF导出作为下一阶段扩展。")
        if st.button("记录报告导出占位操作", key="llm_export"):
            write_log("llm_report", "记录报告导出占位操作")
            st.success("已记录导出占位操作。")
