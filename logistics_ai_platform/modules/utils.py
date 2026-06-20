from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "project_root": PROJECT_ROOT,
    "raw_data": PROJECT_ROOT / "data" / "raw",
    "cleaned_data": PROJECT_ROOT / "data" / "cleaned",
    "processed_data": PROJECT_ROOT / "data" / "processed",
    "sample_docs": PROJECT_ROOT / "data" / "sample_docs",
    "workflows": PROJECT_ROOT / "workflows",
    "reports": PROJECT_ROOT / "outputs" / "reports",
    "charts": PROJECT_ROOT / "outputs" / "charts",
    "logs": PROJECT_ROOT / "outputs" / "logs",
    "workflow_results": PROJECT_ROOT / "outputs" / "workflow_results",
    "assets": PROJECT_ROOT / "assets",
}


def get_project_paths() -> dict[str, Path]:
    """Return common project paths."""
    return PATHS


def ensure_project_dirs() -> None:
    """Create runtime directories if they do not exist."""
    for key, path in PATHS.items():
        if key == "project_root":
            continue
        path.mkdir(parents=True, exist_ok=True)


def load_css() -> None:
    """Load the optional CSS file for Streamlit pages."""
    css_path = PATHS["assets"] / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def write_log(module_name: str, message: str) -> None:
    """Append one line to the platform runtime log."""
    ensure_project_dirs()
    log_file = PATHS["logs"] / "platform.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old_content = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    log_file.write_text(old_content + f"{timestamp}\t{module_name}\t{message}\n", encoding="utf-8")


def list_files(folder: Path) -> pd.DataFrame:
    """Return a compact file list for display."""
    if not folder.exists():
        return pd.DataFrame(columns=["file_name", "size_kb", "modified_time"])

    rows = []
    for file_path in sorted(folder.glob("*")):
        if file_path.is_file():
            stat = file_path.stat()
            rows.append(
                {
                    "file_name": file_path.name,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    return pd.DataFrame(rows, columns=["file_name", "size_kb", "modified_time"])


def read_recent_logs(limit: int = 12) -> list[str]:
    """Read the latest log lines."""
    log_file = PATHS["logs"] / "platform.log"
    if not log_file.exists():
        return []
    lines = log_file.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


def render_workflow_steps(active_step: str = "骨架占位") -> None:
    """Render the shared module process required for the competition demo."""
    steps = ["导入数据", "参数设置", "运行处理", "查看日志", "展示结果", "导出结果"]
    st.caption(f"当前阶段：{active_step}")
    cols = st.columns(len(steps))
    for index, step in enumerate(steps):
        with cols[index]:
            st.markdown(f"**{index + 1}. {step}**")


def render_file_status(title: str, folders: Iterable[Path]) -> None:
    """Show file status tables for selected folders."""
    st.markdown(f"#### {title}")
    for folder in folders:
        st.write(f"目录：`{folder.relative_to(PROJECT_ROOT)}`")
        files = list_files(folder)
        if files.empty:
            st.info("当前目录暂无文件。")
        else:
            st.dataframe(files, use_container_width=True, hide_index=True)
