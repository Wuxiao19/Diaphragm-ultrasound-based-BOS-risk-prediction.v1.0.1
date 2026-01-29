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
# Run detection
# ============================================================

st.subheader("2. Run inference")
run_button = st.button("Start detection", type="primary")


def clear_logs():
    st.session_state.log_messages = []


def clear_session_results():
    """Clear last run results from session_state (UI only)."""
    for k in [
        "last_results_df",
        "last_output_dir",
        "last_is_folder",
        "last_missing_df",
        "last_missing_csv_path",
        "last_temp_dirs",
    ]:
        if k in st.session_state:
            del st.session_state[k]


def cleanup_temp_dirs():
    """Delete temporary upload directories created in this session."""
    temp_dirs = st.session_state.get("last_temp_dirs", [])
    for d in temp_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    st.session_state["last_temp_dirs"] = []
    # Optional: free CUDA cache if available (helps long-running GPU servers)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def prune_detect_runs(keep_last: int) -> None:
    """
    Remove older detect/runX directories to avoid unbounded disk growth.
    keep_last=0 means "do not remove".
    """
    if keep_last <= 0:
        return
    detect_dir = Path.cwd() / "detect"
    if not detect_dir.exists():
        return
    runs = []
    for p in detect_dir.iterdir():
        if p.is_dir() and p.name.startswith("run"):
            try:
                n = int(p.name[3:])
                runs.append((n, p))
            except Exception:
                continue
    if len(runs) <= keep_last:
        return
    runs.sort(key=lambda x: x[0])
    to_delete = runs[: max(0, len(runs) - keep_last)]
    for _, p in to_delete:
        shutil.rmtree(p, ignore_errors=True)


if run_button:
    # Basic input validation
    try:
        if input_mode == "single":
            if not ("b_file" in locals() and b_file) or not (
                "m_file" in locals() and m_file
            ):
                st.error("Please upload both one B-mode image and one M-mode image.")
                st.stop()
        else:
            if not ("b_files" in locals() and b_files) or len(b_files) == 0:
                st.error("Please upload at least one B-mode image for batch mode.")
                st.stop()
            if not ("m_files" in locals() and m_files) or len(m_files) == 0:
                st.error("Please upload at least one M-mode image for batch mode.")
                st.stop()

        clear_logs()
        # Clear last results (avoid confusion from UI showing multiple runs)
        clear_session_results()

        with st.spinner("Loading models and running detection, please wait..."):
            pipeline = get_pipeline()

            # Save uploaded files and call original pipeline
            # Important: each run uses a unique sub-directory to avoid mixing with history
            use_csv = False
            if input_mode == "single":
                b_path = save_uploaded_file(b_file, _new_run_subdir("B_single"))
                m_path = save_uploaded_file(m_file, _new_run_subdir("M_single"))
                is_folder = False
            else:
                b_path = save_uploaded_files_as_folder(b_files, _new_run_subdir("B_folder"))
                m_path = save_uploaded_files_as_folder(m_files, _new_run_subdir("M_folder"))
                is_folder = True

            # Call the same core logic as local GUI
            output_dir, results_df = pipeline.run(
                b_input=b_path,
                m_input=m_path,
                is_folder=is_folder,
                use_csv=use_csv,
            )

        # Save this run's results into session_state (for later display/download)
        st.session_state["last_output_dir"] = output_dir
        st.session_state["last_results_df"] = results_df
        st.session_state["last_is_folder"] = is_folder
        # Save temp dirs for cleanup (using parent for single-image mode)
        st.session_state["last_temp_dirs"] = list(
            {
                str(Path(b_path).parent) if not is_folder else str(Path(b_path)),
                str(Path(m_path).parent) if not is_folder else str(Path(m_path)),
            }
        )
        # 保存路径供 Agent 模式使用（调用全局的 to_relative_path 工具函数）
        if is_folder:
            st.session_state["last_b_folder"] = to_relative_path(b_path)
            st.session_state["last_m_folder"] = to_relative_path(m_path)
            st.session_state["last_b_path"] = None
            st.session_state["last_m_path"] = None
        else:
            st.session_state["last_b_path"] = to_relative_path(b_path)
            st.session_state["last_m_path"] = to_relative_path(m_path)
            st.session_state["last_b_folder"] = None
            st.session_state["last_m_folder"] = None

        # 注意：不要立刻删除 uploaded_inputs 里的临时文件，
        # 否则后面的 Agent 模式就找不到这些图片了。
        # cleanup_temp_dirs()
        # 仍然保留对 detect/runX 结果目录的清理，避免无限增长
        prune_detect_runs(KEEP_LAST_RUNS)

    except Exception as e:
        st.error(f"Error occurred during detection: {e}")


# ============================================================
# Show last run results within the same session (Streamlit rerun)
# ============================================================
if "last_results_df" in st.session_state and "last_output_dir" in st.session_state:
    results_df = st.session_state["last_results_df"]
    output_dir = st.session_state["last_output_dir"]
    is_folder = st.session_state.get("last_is_folder", False)

    st.success("Detection completed!")

    # Logs
    with st.expander("Show processing logs", expanded=False):
        if st.session_state.log_messages:
            st.text("".join(st.session_state.log_messages))
        else:
            st.write("No logs.")

    # Show prediction results table
    st.subheader("3. Detection result preview")

    # Show key result columns if they exist; otherwise show all columns
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

    # Download result CSV
    st.download_button(
        label="Download result CSV",
        data=results_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="detect_result_streamlit.csv",
        mime="text/csv",
    )


# ============================================================
# 全局：Qwen Agent 模式（不再强制依赖 Run inference）
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


    # Missing modality samples
    missing_df = st.session_state.get("last_missing_df")
    missing_csv_path = os.path.join(output_dir, "missing_modality_samples.csv")

    # If we don't have it in session_state but the CSV exists on disk, try to load it
    if (missing_df is None or missing_df.empty) and os.path.exists(missing_csv_path):
        try:
            missing_df = pd.read_csv(missing_csv_path)
            st.session_state["last_missing_df"] = missing_df
        except Exception:
            missing_df = None

    if missing_df is not None and not missing_df.empty:
        st.warning(
            "Some samples are missing B or M modality and were excluded from prediction. "
            "Download the missing list to verify your data."
        )
        with st.expander("Show list of samples with missing modality (downloadable)", expanded=False):
            st.dataframe(missing_df)
            st.download_button(
                label="Download missing modality CSV",
                data=missing_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="missing_modality_samples.csv",
                mime="text/csv",
            )

    # Recheck detection for batch patients
    if is_folder and "merged_filename" in results_df.columns:
        pattern = re.compile(r"(\d{2}-\d{2}-\d{2})-(C\d{3}|B\d{3}|P\d{3})")

        def _parse_merged(name: str):
            m = pattern.match(str(name))
            if m:
                return m.group(1), m.group(2)
            return None, None

        temp = results_df.copy()
        temp[["date", "patient_id"]] = temp["merged_filename"].apply(
            lambda x: pd.Series(_parse_merged(x))
        )
        temp = temp.dropna(subset=["date", "patient_id"])

        if not temp.empty:
            date_counts = temp.groupby("patient_id")["date"].nunique()
            recheck_ids = date_counts[date_counts > 1].index.tolist()

            recheck_df = temp[temp["patient_id"].isin(recheck_ids)].copy()

            if not recheck_df.empty:
                st.warning(
                    "Recheck detected: some patient IDs have examinations on different dates. "
                    "You can download a CSV containing only these recheck cases."
                )
                st.download_button(
                    label="Download recheck result CSV",
                    data=recheck_df[key_cols].to_csv(
                        index=False, encoding="utf-8-sig"
                    ),
                    file_name="recheck_result.csv",
                    mime="text/csv",
                )


st.markdown("---")
st.caption(
    "Developed by AlMSLab"
)
