from __future__ import annotations

import streamlit as st

from modules import workflow_center
from modules.utils import ensure_project_dirs, get_project_paths, load_css


APP_TITLE = "物流企业AI智能运营决策平台"

PAGE_OPTIONS = [
    "模块首页",
    "数据源管理",
    "数据质量概览",
    "数据预处理工作流",
    "数据连接模块",
]


def load_module_css(paths) -> None:
    css_path = paths["assets"] / "module1.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_project_dirs()
    paths = get_project_paths()
    load_css()
    load_module_css(paths)
    if "quality_report_cleaned_on_start" not in st.session_state:
        workflow_center.cleanup_quality_reports(paths)
        st.session_state["quality_report_cleaned_on_start"] = True

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <span>Module 01</span>
                <strong>数据工程中心</strong>
                <p>学生1：数据工程师</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_page = st.radio("功能菜单", PAGE_OPTIONS, label_visibility="collapsed")
        st.divider()
        st.markdown("**当前模块**")
        st.write("可定制数据清洗流程中心")
        st.caption("数据导入、质量检查、流程化清洗与指标连接。")

    st.markdown(
        """
        <div class="topbar">
            <div>
                <span class="topbar-label">Logistics AI Operations / Data Engineering</span>
                <strong>物流企业AI智能运营决策平台</strong>
            </div>
            <span class="role-chip">学生1：数据工程师</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    workflow_center.render_page(selected_page, paths)


if __name__ == "__main__":
    main()
