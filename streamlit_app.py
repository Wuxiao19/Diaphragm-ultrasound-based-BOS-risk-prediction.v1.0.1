import os
import re
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from integrated_detection_gui_ET import DetectionPipeline
from ultrasound_agent import (
    qwen_explain_detection_sync,
    run_qwen_agent,
)
import asyncio


# ============================================================
# Streamlit basic page configuration
# ============================================================

st.set_page_config(
    page_title="Diaphragm Ultrasound Analysis System",
    page_icon="🩺",
    layout="wide",
)

st.title("Diaphragm Ultrasound Analysis System")
st.markdown(
    """  
Upload **B-mode** and **M-mode** diaphragm ultrasound images
for one patient (single exam) or for multiple patients (batch exams).
The system will automatically perform: feature extraction → feature reduction
→ feature fusion → ExtraTrees-based binary classification.

**Filename convention (IMPORTANT):**
- filenames must contain `YY-MM-DD-<ID>` pattern, e.g. `24-05-01-C001_xxx`
"""
)


# ============================================================
# Cache DetectionPipeline instance (avoid re-loading models)
# ============================================================

@st.cache_resource(show_spinner=True)
def get_pipeline():
    """
    Create and cache a DetectionPipeline instance.
    We do not modify internal logic of integrated_detection_gui_ET.DetectionPipeline,
    only reuse it here.
    """

    # Use simple callback to accumulate logs into session_state for display
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []

    def gui_callback(msg: str):
        st.session_state.log_messages.append(msg)

    pipeline = DetectionPipeline(gui_callback=gui_callback)
    pipeline.load_models()
    return pipeline


# ============================================================
# Helper functions: handle uploaded files and temp dirs
# ============================================================

def ensure_upload_dir() -> Path:
    """Ensure base directory for temporary uploaded files exists."""
    base_dir = Path.cwd() / "uploaded_inputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _clear_dir(path: Path) -> None:
    """Remove and recreate a directory, ignoring errors if it does not exist."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _new_run_subdir(prefix: str) -> str:
    """
    Create a unique sub-directory name for this run,
    so that different users/runs do not interfere with each other.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def save_uploaded_file(uploaded_file, subdir: str) -> str:
    """
    Save a single uploaded file to disk and return its local path.
    """
    upload_root = ensure_upload_dir()
    target_dir = upload_root / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return str(file_path)


def save_uploaded_files_as_folder(uploaded_files, subdir: str) -> str:
    """
    Save multiple uploaded files to one sub-directory,
    simulating a "folder" input. Return that directory path.
    """
    upload_root = ensure_upload_dir()
    target_dir = upload_root / subdir
    _clear_dir(target_dir)

    for uf in uploaded_files:
        file_path = target_dir / uf.name
        with open(file_path, "wb") as f:
            f.write(uf.read())
    return str(target_dir)


def to_relative_path(abs_path: str) -> str:
    """
    将绝对路径转换为相对于项目根目录的相对路径，主要用于把本地保存的
    uploaded_inputs 路径，转换成 MCP 服务器也能识别的形式。
    """
    try:
        abs_path_obj = Path(abs_path)
        # 优先尝试：相对于当前工作目录的相对路径
        try:
            rel_path = abs_path_obj.relative_to(Path.cwd())
            return str(rel_path)
        except ValueError:
            # 如果无法直接 relative_to，则尝试截取 "uploaded_inputs" 之后的部分
            parts = abs_path_obj.parts
            if "uploaded_inputs" in parts:
                idx = parts.index("uploaded_inputs")
                rel_parts = parts[idx:]
                return str(Path(*rel_parts))
            # 最后兜底：只返回文件名
            return abs_path_obj.name
    except Exception:
        # 任何异常都直接返回原始路径，避免中断流程
        return abs_path


# Keep at most this number of recent detect/runX directories (older ones are removed).
KEEP_LAST_RUNS = 20


# ============================================================
# Sidebar: input mode
# ============================================================

st.sidebar.header("Input settings")

input_mode = st.sidebar.radio(
    "Input mode",
    options=["single", "folder"],
    format_func=lambda x: "Single patient (one B + one M image)"
    if x == "single"
    else "Batch patients (multiple B- and M-mode images)",
)


# ============================================================
# Main area: file upload
# ============================================================

st.subheader("1. Upload input data")
st.caption(
    "File naming rule: each filename must start with `YY-MM-DD-<ID>`, "
    "e.g. `24-05-01-C001_xxx.png`. The same patient ID on the same date "
    "will be merged as one exam."
)

col_b, col_m = st.columns(2)

with col_b:
    if input_mode == "single":
        b_file = st.file_uploader(
            "Upload B-mode image (single)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="b_image_single",
        )
    else:
        b_files = st.file_uploader(
            "Upload B-mode images (multiple files allowed, treated as one folder)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="b_image_folder",
        )

with col_m:
    if input_mode == "single":
        m_file = st.file_uploader(
            "Upload M-mode image (single)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="m_image_single",
        )
    else:
        m_files = st.file_uploader(
            "Upload M-mode images (multiple files allowed, treated as one folder)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="m_image_folder",
        )


st.markdown("---")
# ============================================================
# 全局：Qwen Agent 模式（不再强制依赖 Run inference）
# ============================================================
st.subheader("2. AI 检测与解读（Qwen3-8B Agent）")
st.caption(
    "说明：本模式下，你只需要在上方上传膈肌 B 模式和 M 模式超声图像，"
    "Qwen 会自动调用后端的检测流水线（MCP 工具），完成特征提取和风险预测，然后给出中文解读。"
    "该解读不能替代医生最终诊断。"
)

with st.expander("点击展开：配置 Qwen（SiliconFlow/OpenAI 兼容接口）", expanded=False):
    qwen_api_key = st.text_input(
        "Qwen API Key（建议填到环境变量 QWEN_API_KEY；这里也可临时输入）",
        type="password",
        value=os.getenv("QWEN_API_KEY", ""),
    )
    qwen_base_url = st.text_input(
        "Base URL（保持默认即可）",
        value=os.getenv("QWEN_BASE_URL", "https://api.siliconflow.cn/v1"),
    )
    qwen_model = st.text_input(
        "Model（保持默认即可）",
        value=os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B"),
    )

st.info("🤖 **Agent 模式**：直接基于你上传的图像，调用后端检测工具并生成完整分析，无需先点击 Run inference。")

if st.button("🚀 启动 Qwen Agent（自动调用检测工具）", type="primary"):
    if not qwen_api_key.strip():
        st.error("请先填写 Qwen API Key（或在系统环境变量里设置 QWEN_API_KEY）。")
    else:
        # 根据当前输入模式，准备传给 Agent 的本地路径
        b_path_for_agent = None
        m_path_for_agent = None
        b_folder_for_agent = None
        m_folder_for_agent = None

        try:
            if input_mode == "single":
                if not ("b_file" in locals() and b_file) or not ("m_file" in locals() and m_file):
                    st.error("请先在上方上传一张 B 模式和一张 M 模式图片。")
                    st.stop()

                # 保存单张图片到本地 uploaded_inputs 子目录
                b_abs = save_uploaded_file(b_file, _new_run_subdir("B_single_agent"))
                m_abs = save_uploaded_file(m_file, _new_run_subdir("M_single_agent"))

                b_path_for_agent = to_relative_path(b_abs)
                m_path_for_agent = to_relative_path(m_abs)

            else:  # folder 模式
                if not ("b_files" in locals() and b_files) or len(b_files) == 0:
                    st.error("请先上传至少一张 B 模式图片（批量模式）。")
                    st.stop()
                if not ("m_files" in locals() and m_files) or len(m_files) == 0:
                    st.error("请先上传至少一张 M 模式图片（批量模式）。")
                    st.stop()

                b_abs_dir = save_uploaded_files_as_folder(b_files, _new_run_subdir("B_folder_agent"))
                m_abs_dir = save_uploaded_files_as_folder(m_files, _new_run_subdir("M_folder_agent"))

                b_folder_for_agent = to_relative_path(b_abs_dir)
                m_folder_for_agent = to_relative_path(m_abs_dir)

            with st.spinner("🤖 Qwen Agent 正在工作：调用检测工具并生成分析..."):
                if b_path_for_agent and m_path_for_agent:
                    agent_result = asyncio.run(
                        run_qwen_agent(
                            b_image_path=b_path_for_agent,
                            m_image_path=m_path_for_agent,
                            api_key=qwen_api_key.strip(),
                            base_url=qwen_base_url.strip(),
                            model=qwen_model.strip(),
                        )
                    )
                elif b_folder_for_agent and m_folder_for_agent:
                    agent_result = asyncio.run(
                        run_qwen_agent(
                            b_folder_path=b_folder_for_agent,
                            m_folder_path=m_folder_for_agent,
                            api_key=qwen_api_key.strip(),
                            base_url=qwen_base_url.strip(),
                            model=qwen_model.strip(),
                        )
                    )
                else:
                    raise ValueError("无法确定是单组还是批量模式，请检查上传的文件。")

            st.success("✅ Qwen Agent 分析完成！")

            # 显示工具调用记录
            with st.expander("🔧 查看：Qwen 调用了哪些工具", expanded=False):
                if agent_result["tool_calls"]:
                    for i, tc in enumerate(agent_result["tool_calls"], 1):
                        st.write(f"**工具 {i}**：`{tc['name']}`")
                        st.json(tc["arguments"])
                else:
                    st.write("（本次未调用工具）")

            # 显示工具返回结果（调试用）
            with st.expander("📊 查看：工具返回的原始 JSON（调试用）", expanded=False):
                for name, res in agent_result["tool_results"].items():
                    st.write(f"**{name}** 返回结果：")
                    st.json(res)

            # 显示 Qwen 最终回答
            st.markdown("---")
            st.markdown("### 💬 Qwen Agent 的完整分析")
            st.markdown(agent_result["final_response"])

        except Exception as e:
            st.error(f"❌ Qwen Agent 运行失败：{e}")
            import traceback
            with st.expander("查看详细错误信息", expanded=False):
                st.code(traceback.format_exc())


st.markdown("---")
st.caption(
    "Developed by AlMSLab"
)


