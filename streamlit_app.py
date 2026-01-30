import os
import re
from pathlib import Path
import shutil
import uuid
from random import randint

import numpy as np
import pandas as pd
import streamlit as st

from ultrasound_agent import run_qwen_agent, _build_detection_summary_from_tool_result
import asyncio
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from session_state import get_session_state, persist, load_widget_state


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

# 初始化 session_state
if "detect_output_dir" not in st.session_state:
    st.session_state["detect_output_dir"] = None


def _clear_session_results():
    """清除上一次运行的结果，避免 UI 显示混乱。"""
    st.session_state.pop("agent_result", None)
    st.session_state["detect_output_dir"] = None


def _render_agent_result(ar: dict) -> None:
    """Render agent result stored in session_state or returned from run_qwen_agent.

    This is separated so the UI remains visible across Streamlit reruns (e.g. after
    clicking download_button) because we read from st.session_state.
    """
    if not ar:
        return

    with st.expander("🔧 查看：Qwen 调用了哪些工具", expanded=False):
        if ar.get("tool_calls"):
            for i, tc in enumerate(ar["tool_calls"], 1):
                st.write(f"**工具 {i}**：`{tc['name']}`")
                st.json(tc["arguments"])
        else:
            st.write("（本次未调用工具）")

    with st.expander("📊 查看：工具返回的原始 JSON（调试用）", expanded=False):
        if ar.get("tool_results"):
            for name, res in ar.get("tool_results", {}).items():
                st.write(f"**{name}** 返回结果：")
                st.json(res)
        else:
            st.write("（无工具返回结果）")

    st.markdown("### 💬 Qwen Agent 的完整分析")
    detection_summary = None
    if isinstance(ar.get("tool_results"), dict):
        for name, res in ar.get("tool_results", {}).items():
            if isinstance(res, dict) and res:
                try:
                    detection_summary = _build_detection_summary_from_tool_result(res)
                    break
                except Exception:
                    detection_summary = None

    if detection_summary:
        cols = st.columns([1, 1, 1])
        cols[0].metric("样本数量", detection_summary.get("total_samples", 0))
        cols[1].metric("平均风险概率", f"{detection_summary.get('average_probability', 0.0):.3f}")
        cols[2].metric("复检患者数", len(detection_summary.get("recheck_patients", [])))

        st.markdown("**样本详情（表格）**")
        items_df = pd.DataFrame(detection_summary.get("items", []))
        if not items_df.empty:
            st.dataframe(items_df)

        high_risk = items_df[items_df["risk_probability"] > 0.7] if not items_df.empty else pd.DataFrame()
        if not high_risk.empty:
            st.warning("检测到高风险患者（risk_probability > 0.7）：")
            st.table(high_risk[["merged_key", "patient_id", "date", "risk_probability"]])

        if detection_summary.get("recheck_patients"):
            with st.expander("复检患者（同一患者在不同日期的随访）", expanded=False):
                for rp in detection_summary.get("recheck_patients", []):
                    st.write(f"患者 ID：{rp.get('patient_id')}")
                    st.write("检查日期：" + ", ".join(rp.get("exam_dates", [])))
                    st.dataframe(pd.DataFrame(rp.get("visits", [])))

        if detection_summary.get("missing_modality_summary"):
            ms = detection_summary["missing_modality_summary"]
            st.info(f"缺失模态样本数：{ms.get('total_missing_samples', 0)}")
            if ms.get("missing_by_type"):
                st.write("按缺失类型统计：")
                st.json(ms.get("missing_by_type"))

    final_text = ar.get("final_response", "")
    if final_text:
        st.markdown("---")
        st.markdown("#### 原始模型文本输出")
        st.markdown(final_text)
    else:
        st.info("（模型未生成文本；请查看调试信息）")


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

# Ensure dynamic key update for file uploaders
state = get_session_state(b_widget_key=str(randint(1000, 100000000)), m_widget_key=str(randint(1000, 100000000)))

with col_b:
    if input_mode == "single":
        b_file = st.file_uploader(
            "Upload B-mode image (single)",
            type=["jpg", "jpeg", "png", "bmp"],
            key=state.b_widget_key,
        )
    else:
        b_files = st.file_uploader(
            "Upload B-mode images (multiple files allowed, treated as one folder)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key=state.b_widget_key,
        )
    if st.button("Clear B-mode uploads"):
        state.b_widget_key = str(randint(1000, 100000000))
        state.sync()
        st.experimental_rerun()  # Force rerun to refresh uploader

with col_m:
    if input_mode == "single":
        m_file = st.file_uploader(
            "Upload M-mode image (single)",
            type=["jpg", "jpeg", "png", "bmp"],
            key=state.m_widget_key,
        )
    else:
        m_files = st.file_uploader(
            "Upload M-mode images (multiple files allowed, treated as one folder)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key=state.m_widget_key,
        )
    if st.button("Clear M-mode uploads"):
        state.m_widget_key = str(randint(1000, 100000000))
        state.sync()
        st.experimental_rerun()  # Force rerun to refresh uploader

# Ensure variables are defined before use
b_file, b_files, m_file, m_files = None, None, None, None

# Clear previous uploads when new files are uploaded
if "b_image_single" in st.session_state and b_file is not None:
    del st.session_state["b_image_single"]
if "b_image_folder" in st.session_state and b_files:
    del st.session_state["b_image_folder"]
if "m_image_single" in st.session_state and m_file is not None:
    del st.session_state["m_image_single"]
if "m_image_folder" in st.session_state and m_files:
    del st.session_state["m_image_folder"]

# ============================================================
# 全局：Qwen Agent 模式
# ============================================================
st.markdown("---")
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
        # 清除上一次运行的结果
        _clear_session_results()
        
        # 根据当前输入模式，准备传给 Agent 的本地路径
        b_path_for_agent = None
        m_path_for_agent = None
        b_folder_for_agent = None
        m_folder_for_agent = None

        try:
            # 保险：再次确保删除 uploaded_inputs 下的所有旧上传子目录
            # （on_change 中应该已删除，但为了安全再做一次）
            try:
                upload_root = ensure_upload_dir()
                for child in upload_root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass

            if input_mode == "single":
                if not ("b_file" in locals() and b_file) or not ("m_file" in locals() and m_file):
                    st.error("请先在上方上传一张 B 模式和一张 M 模式图片。")
                    st.stop()

                # 保存单张图片到本地 uploaded_inputs 子目录
                b_abs = save_uploaded_file(b_file, _new_run_subdir("B_single_agent"))
                m_abs = save_uploaded_file(m_file, _new_run_subdir("M_single_agent"))

                # 直接传递绝对路径，避免 MCP 在 uploaded_inputs 根目录下混合旧文件
                b_path_for_agent = str(Path(b_abs).resolve())
                m_path_for_agent = str(Path(m_abs).resolve())

            else:  # folder 模式
                if not ("b_files" in locals() and b_files) or len(b_files) == 0:
                    st.error("请先上传至少一张 B 模式图片（批量模式）。")
                    st.stop()
                if not ("m_files" in locals() and m_files) or len(m_files) == 0:
                    st.error("请先上传至少一张 M 模式图片（批量模式）。")
                    st.stop()

                b_abs_dir = save_uploaded_files_as_folder(b_files, _new_run_subdir("B_folder_agent"))
                m_abs_dir = save_uploaded_files_as_folder(m_files, _new_run_subdir("M_folder_agent"))

                # 直接传递绝对目录，避免 MCP 搜索到 uploaded_inputs 其它旧文件
                b_folder_for_agent = str(Path(b_abs_dir).resolve())
                m_folder_for_agent = str(Path(m_abs_dir).resolve())

            # 在开始新一次 Agent 运行前，清除上一次的显示（仅在真正开始运行时）
            st.session_state.pop("agent_result", None)

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

            # 将 agent_result 清理为纯可序列化结构后保存到 session_state，
            # 避免包含非可序列化对象（如 Path、DataFrame 或连接句柄）导致
            # Streamlit 在 rerun 时无法持久化 session_state 的问题。
            def _sanitize_agent_result(ar):
                if not isinstance(ar, dict):
                    return ar
                out = {}
                for k, v in ar.items():
                    try:
                        # pandas DataFrame -> records
                        if hasattr(v, "to_dict") and callable(getattr(v, "to_dict")):
                            out[k] = v.to_dict()
                        # numpy types
                        elif isinstance(v, (np.integer, np.floating)):
                            out[k] = v.item()
                        elif isinstance(v, (list, tuple)):
                            new_list = []
                            for e in v:
                                if hasattr(e, "to_dict"):
                                    new_list.append(e.to_dict())
                                else:
                                    new_list.append(e)
                            out[k] = new_list
                        else:
                            out[k] = v
                    except Exception:
                        # 兜底，转换为字符串表示
                        try:
                            out[k] = json.loads(json.dumps(v, default=str))
                        except Exception:
                            out[k] = str(v)
                return out

            sanitized_agent_result = _sanitize_agent_result(agent_result)

            # 持久化 agent 结果到 session_state，这样下载等操作不会清除显示
            st.session_state["agent_result"] = sanitized_agent_result
            # 注：渲染逻辑在页面全局最后的代码块中，避免重复渲染

            # ====================================================
            # 从 MCP 工具结果中解析 detect_output_dir，加载并导出 CSV
            # ====================================================
            detect_output_dir = None
            ar = st.session_state.get("agent_result")
            for name, res in (ar.get("tool_results", {}) if ar else {}).items():
                if isinstance(res, dict) and "detect_output_dir" in res:
                    detect_output_dir = res["detect_output_dir"]
                    break
            
            # 保存到 session_state，这样下载按钮可以在任何 rerun 中访问它
            if detect_output_dir:
                st.session_state["detect_output_dir"] = detect_output_dir

        except Exception as e:
            st.error(f"❌ Qwen Agent 运行失败：{e}")
            import traceback
            with st.expander("查看详细错误信息", expanded=False):
                st.code(traceback.format_exc())

# 如果 session 中存在上一次 agent 的结果，始终渲染它（保证在任何 rerun 后都可见）
if st.session_state.get("agent_result"):
    try:
        _render_agent_result(st.session_state.get("agent_result"))
    except Exception:
        # 渲染失败不应阻塞主流程，保证页面其它部分可用
        pass


# ====================================================
# 全局：显示 CSV 下载和预览（如果有检测结果）
# ====================================================
detect_output_dir = st.session_state.get("detect_output_dir")
if isinstance(detect_output_dir, str) and detect_output_dir:
    result_csv_path = os.path.join(detect_output_dir, "detect_result.csv")
    if os.path.exists(result_csv_path):
        try:
            results_df = pd.read_csv(result_csv_path)

            st.markdown("---")
            st.markdown("### 📊 检测结果预览（来自 MCP 流水线）")
            key_cols = [
                "merged_filename",
                "b_filename",
                "m_filename",
                "risk_probability",
                "prediction",
                "prediction_label",
            ]
            show_cols = [c for c in key_cols if c in results_df.columns]
            st.dataframe(results_df[show_cols] if show_cols else results_df)

            st.download_button(
                label="下载检测结果 CSV",
                data=results_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="detect_result.csv",
                mime="text/csv",
            )

            # 缺失模态样本（如果存在）
            missing_csv_path = os.path.join(detect_output_dir, "missing_modality_samples.csv")
            if os.path.exists(missing_csv_path):
                try:
                    missing_df = pd.read_csv(missing_csv_path)
                except Exception:
                    missing_df = None

                if missing_df is not None and not missing_df.empty:
                    st.warning(
                        "部分样本缺失 B 或 M 模态，因此未参与最终预测。"
                        "你可以下载缺失样本列表进行排查。"
                    )
                    with st.expander(
                        "Show list of samples with missing modality (downloadable)",
                        expanded=False,
                    ):
                        st.dataframe(missing_df)
                        st.download_button(
                            label="下载缺失模态 CSV",
                            data=missing_df.to_csv(index=False, encoding="utf-8-sig"),
                            file_name="missing_modality_samples.csv",
                            mime="text/csv",
                        )
        except Exception:
            # 如果读取失败，不影响主流程，只是不显示表格和下载按钮
            pass

st.markdown("---")
st.caption(
    "Developed by AlMSLab"
)


