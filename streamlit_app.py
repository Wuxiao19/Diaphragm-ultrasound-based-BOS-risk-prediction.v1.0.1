import os
import re
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from ultrasound_agent import run_qwen_agent, _build_detection_summary_from_tool_result
import asyncio
import json


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
    Convert an absolute path to a project-root-relative path. This is mainly used
    to turn local uploaded_inputs paths into a form that the MCP server can resolve.
    """
    try:
        abs_path_obj = Path(abs_path)
        # Prefer a path relative to the current working directory
        try:
            rel_path = abs_path_obj.relative_to(Path.cwd())
            return str(rel_path)
        except ValueError:
            # If relative_to fails, try slicing after "uploaded_inputs"
            parts = abs_path_obj.parts
            if "uploaded_inputs" in parts:
                idx = parts.index("uploaded_inputs")
                rel_parts = parts[idx:]
                return str(Path(*rel_parts))
            return abs_path_obj.name
    except Exception:
        # Return the original path on any exception to avoid breaking the flow
        return abs_path


# Keep at most this number of recent detect/runX directories (older ones are removed).
KEEP_LAST_RUNS = 20

# Initialize session_state
if "last_upload_key" not in st.session_state:
    st.session_state["last_upload_key"] = None
if "detect_output_dir" not in st.session_state:
    st.session_state["detect_output_dir"] = None


def _on_file_uploader_change(mode: str) -> None:
    """Clear previous detection results when the user uploads new files."""
    st.session_state.pop("agent_result", None)
    st.session_state["detect_output_dir"] = None


def _render_agent_result(ar: dict) -> None:
    """Render agent result stored in session_state or returned from run_qwen_agent.

    This is separated so the UI remains visible across Streamlit reruns (e.g. after
    clicking download_button) because we read from st.session_state.
    """
    if not ar:
        return

    with st.expander("🔧 View: Tools called by Qwen", expanded=False):
        if ar.get("tool_calls"):
            for i, tc in enumerate(ar["tool_calls"], 1):
                st.write(f"**Tool {i}**: `{tc['name']}`")
                st.json(tc["arguments"])
        else:
            st.write("(No tools called this run)")

    with st.expander("📊 View: Raw JSON returned by tools (debug)", expanded=False):
        if ar.get("tool_results"):
            for name, res in ar.get("tool_results", {}).items():
                st.write(f"**{name}** result:")
                st.json(res)
        else:
            st.write("(No tool results)")

    st.markdown("### 💬 Qwen Agent full analysis")
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
        def _find_image_path(filename: str) -> str | None:
            if not filename:
                return None
            # 1) Search under uploaded_inputs
            try:
                upload_root = ensure_upload_dir()
                for p in upload_root.rglob(filename):
                    if p.is_file():
                        return str(p)
            except Exception:
                pass

            # 2) Search under detect_output_dir (if available)
            try:
                detect_dir = st.session_state.get("detect_output_dir")
                if isinstance(detect_dir, str) and detect_dir:
                    detect_path = Path(detect_dir)
                    if detect_path.exists():
                        for p in detect_path.rglob(filename):
                            if p.is_file():
                                return str(p)
            except Exception:
                pass

            return None

        cols = st.columns([1, 1])
        cols[0].metric("Samples", detection_summary.get("total_samples", 0))
        cols[1].metric("Recheck patients", len(detection_summary.get("recheck_patients", [])))

        st.markdown("**Sample details (table)**")
        items_df = pd.DataFrame(detection_summary.get("items", []))
        if not items_df.empty:
            display_df = items_df[["patient_id", "date", "risk_probability"]].copy()
            display_df = display_df.rename(
                columns={
                    "patient_id": "Patient_id",
                    "date": "date",
                    "risk_probability": "risk_probability",
                }
            )

            def _risk_color(val):
                try:
                    v = float(val)
                except Exception:
                    return ""
                if v > 0.6:
                    return "color: #c62828; font-weight: 600;"  # red
                if v < 0.3:
                    return "color: #2e7d32; font-weight: 600;"  # green
                return "color: #ef6c00; font-weight: 600;"  # orange

            styled = display_df.style.format({"risk_probability": "{:.3f}"}).applymap(
                _risk_color, subset=["risk_probability"]
            )
            st.dataframe(styled, use_container_width=True)

        high_risk = items_df[items_df["risk_probability"] > 0.6] if not items_df.empty else pd.DataFrame()
        if not high_risk.empty:
            st.warning("High-risk patients detected (risk_probability > 0.6):")
            st.table(high_risk[["patient_id", "date", "risk_probability"]])

            with st.expander("High-risk patient images (B/M mode)", expanded=False):
                for _, row in high_risk.iterrows():
                    b_name = row.get("b_image") or row.get("b_filename") or ""
                    m_name = row.get("m_image") or row.get("m_filename") or ""
                    b_path = _find_image_path(str(b_name)) if b_name else None
                    m_path = _find_image_path(str(m_name)) if m_name else None

                    st.markdown(
                        f"- Patient {row.get('patient_id')} | {row.get('date')} | risk={float(row.get('risk_probability', 0.0)):.3f}"
                    )
                    img_cols = st.columns(2)
                    with img_cols[0]:
                        st.caption(f"B-mode: {b_name}" if b_name else "B-mode: (not available)")
                        if b_path:
                            st.image(b_path, use_container_width=True)
                        else:
                            st.info("B-mode image not found in uploaded inputs.")
                    with img_cols[1]:
                        st.caption(f"M-mode: {m_name}" if m_name else "M-mode: (not available)")
                        if m_path:
                            st.image(m_path, use_container_width=True)
                        else:
                            st.info("M-mode image not found in uploaded inputs.")

        if detection_summary.get("recheck_patients"):
            with st.expander("Recheck patients (same patient across dates)", expanded=False):
                for rp in detection_summary.get("recheck_patients", []):
                    st.write(f"Patient ID: {rp.get('patient_id')}")
                    st.write("Exam dates: " + ", ".join(rp.get("exam_dates", [])))
                    st.dataframe(pd.DataFrame(rp.get("visits", [])))

        if detection_summary.get("missing_modality_summary"):
            with st.expander("⚠️ Missing modality samples", expanded=False):
                ms = detection_summary["missing_modality_summary"]
                total_missing = ms.get("total_missing_samples", 0)
                missing_by_type = ms.get("missing_by_type") or {}

                st.metric("Missing samples", total_missing)
                if missing_by_type:
                    summary_df = pd.DataFrame(
                        [
                            {"Missing type": k, "Count": v}
                            for k, v in missing_by_type.items()
                        ]
                    )
                    st.markdown("**Missing type summary**")
                    st.table(summary_df)

    final_text = ar.get("final_response", "")
    if final_text:
        st.markdown("---")
        st.markdown("#### Raw model output")
        st.markdown(final_text)
    else:
        st.info("(Model produced no text; check debug info.)")


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
            on_change=lambda: _on_file_uploader_change("single"),
        )
    else:
        b_files = st.file_uploader(
            "Upload B-mode images (multiple files allowed, treated as one folder)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="b_image_folder",
            on_change=lambda: _on_file_uploader_change("folder"),
        )

with col_m:
    if input_mode == "single":
        m_file = st.file_uploader(
            "Upload M-mode image (single)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="m_image_single",
            on_change=lambda: _on_file_uploader_change("single"),
        )
    else:
        m_files = st.file_uploader(
            "Upload M-mode images (multiple files allowed, treated as one folder)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="m_image_folder",
            on_change=lambda: _on_file_uploader_change("folder"),
        )

# ============================================================
# Global: Qwen Agent mode
# ============================================================
st.markdown("---")
st.subheader("2. AI detection and interpretation (Qwen3-8B Agent)")
st.caption(
    "Note: In this mode, simply upload B-mode and M-mode diaphragm ultrasound images above. "
    "Qwen will call the backend detection pipeline (MCP tools), complete feature extraction "
    "and risk prediction, then generate an English interpretation. This cannot replace a doctor's diagnosis."
)

with st.expander("Click to expand: Configure Qwen (SiliconFlow/OpenAI compatible API)", expanded=False):
    qwen_api_key = st.text_input(
        "Qwen API Key (recommended via QWEN_API_KEY env var; you can also enter it here)",
        type="password",
        value=os.getenv("QWEN_API_KEY", ""),
    )
    qwen_base_url = st.text_input(
        "Base URL",
        value=os.getenv("QWEN_BASE_URL", "https://api.siliconflow.cn/v1"),
    )
    qwen_model = st.text_input(
        "Model",
        value=os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B"),
    )

st.info("🤖 **Agent mode**: directly use your uploaded images, call backend detection tools, and generate a full analysis. No need to click Run inference first.")


def _run_agent_safe(**kwargs):
    """Call run_qwen_agent and handle older versions without the language parameter."""
    try:
        return asyncio.run(run_qwen_agent(**kwargs))
    except TypeError as e:
        if "language" in str(e):
            kwargs.pop("language", None)
            return asyncio.run(run_qwen_agent(**kwargs))
        raise



if st.button("🚀 Run Qwen Agent (auto-call detection tools)", type="primary"):
    if not qwen_api_key.strip():
        st.error("Please provide a Qwen API Key (or set QWEN_API_KEY in the environment).")
    else:
        # Prepare local paths for the agent based on input mode
        b_path_for_agent = None
        m_path_for_agent = None
        b_folder_for_agent = None
        m_folder_for_agent = None

        try:
            # Before saving new uploads, remove old uploaded_inputs subdirs
            # to avoid mixing old files into this MCP run.
            try:
                upload_root = ensure_upload_dir()
                for child in upload_root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass

            if input_mode == "single":
                if not ("b_file" in locals() and b_file) or not ("m_file" in locals() and m_file):
                    st.error("Please upload one B-mode and one M-mode image above.")
                    st.stop()

                # Save single images to uploaded_inputs subdirectories
                b_abs = save_uploaded_file(b_file, _new_run_subdir("B_single_agent"))
                m_abs = save_uploaded_file(m_file, _new_run_subdir("M_single_agent"))

                # Pass absolute paths to avoid MCP mixing old files under uploaded_inputs
                b_path_for_agent = str(Path(b_abs).resolve())
                m_path_for_agent = str(Path(m_abs).resolve())

            else:  # folder mode
                if not ("b_files" in locals() and b_files) or len(b_files) == 0:
                    st.error("Please upload at least one B-mode image (batch mode).")
                    st.stop()
                if not ("m_files" in locals() and m_files) or len(m_files) == 0:
                    st.error("Please upload at least one M-mode image (batch mode).")
                    st.stop()

                b_abs_dir = save_uploaded_files_as_folder(b_files, _new_run_subdir("B_folder_agent"))
                m_abs_dir = save_uploaded_files_as_folder(m_files, _new_run_subdir("M_folder_agent"))

                # Pass absolute directories to avoid MCP scanning other old uploads
                b_folder_for_agent = str(Path(b_abs_dir).resolve())
                m_folder_for_agent = str(Path(m_abs_dir).resolve())

            # Clear previous results when starting a new agent run
            st.session_state.pop("agent_result", None)

            with st.spinner("🤖 Qwen Agent is working: calling detection tools and generating analysis..."):
                if b_path_for_agent and m_path_for_agent:
                    agent_result = _run_agent_safe(
                        b_image_path=b_path_for_agent,
                        m_image_path=m_path_for_agent,
                        api_key=qwen_api_key.strip(),
                        base_url=qwen_base_url.strip(),
                        model=qwen_model.strip(),
                    )
                elif b_folder_for_agent and m_folder_for_agent:
                    agent_result = _run_agent_safe(
                        b_folder_path=b_folder_for_agent,
                        m_folder_path=m_folder_for_agent,
                        api_key=qwen_api_key.strip(),
                        base_url=qwen_base_url.strip(),
                        model=qwen_model.strip(),
                    )
                else:
                    raise ValueError("Unable to determine single or batch mode. Please check your uploads.")

            st.success("✅ Qwen Agent analysis complete!")

            # Sanitize agent_result into a serializable structure for session_state
            # to avoid non-serializable objects (Path, DataFrame, handles) on rerun.
            def _sanitize_agent_result(ar):
                if not isinstance(ar, dict):
                    return ar
                out = {}
                for k, v in ar.items():
                    try:
                        # pandas DataFrame -> dict
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
                        # Fallback: convert to string
                        try:
                            out[k] = json.loads(json.dumps(v, default=str))
                        except Exception:
                            out[k] = str(v)
                return out

            sanitized_agent_result = _sanitize_agent_result(agent_result)

            # Persist agent result in session_state so downloads won't clear the view
            st.session_state["agent_result"] = sanitized_agent_result
            # Note: rendering is handled in the global block below to avoid duplicates

            # ====================================================
            # Extract detect_output_dir from tool results for CSV preview/download
            # ====================================================
            detect_output_dir = None
            ar = st.session_state.get("agent_result")
            for name, res in (ar.get("tool_results", {}) if ar else {}).items():
                if isinstance(res, dict) and "detect_output_dir" in res:
                    detect_output_dir = res["detect_output_dir"]
                    break
            
            # Save to session_state so download buttons work across reruns
            if detect_output_dir:
                st.session_state["detect_output_dir"] = detect_output_dir

        except Exception as e:
            st.error(f"❌ Qwen Agent failed: {e}")
            import traceback
            with st.expander("View detailed error info", expanded=False):
                st.code(traceback.format_exc())

# Always render the last agent result from session_state (visible across reruns)
if st.session_state.get("agent_result"):
    try:
        _render_agent_result(st.session_state.get("agent_result"))
    except Exception:
        # Rendering failures should not block the main flow
        pass


# ====================================================
# Global: show CSV preview and download (if results exist)
# ====================================================
detect_output_dir = st.session_state.get("detect_output_dir")
if isinstance(detect_output_dir, str) and detect_output_dir:
    result_csv_path = os.path.join(detect_output_dir, "detect_result.csv")
    if os.path.exists(result_csv_path):
        try:
            results_df = pd.read_csv(result_csv_path)

            st.markdown("---")
            st.markdown("### 📊 Detection results preview (from MCP pipeline)")
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
                label="Download detection results CSV",
                data=results_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="detect_result.csv",
                mime="text/csv",
            )

            # Missing modality samples (if any)
            missing_csv_path = os.path.join(detect_output_dir, "missing_modality_samples.csv")
            if os.path.exists(missing_csv_path):
                try:
                    missing_df = pd.read_csv(missing_csv_path)
                except Exception:
                    missing_df = None

                if missing_df is not None and not missing_df.empty:
                    st.warning(
                        "Some samples are missing B or M modality and were not included in prediction. "
                        "You can download the missing list to review."
                    )
                    with st.expander(
                        "Show list of samples with missing modality (downloadable)",
                        expanded=False,
                    ):
                        st.dataframe(missing_df)
                        st.download_button(
                            label="Download missing modality CSV",
                            data=missing_df.to_csv(index=False, encoding="utf-8-sig"),
                            file_name="missing_modality_samples.csv",
                            mime="text/csv",
                        )
        except Exception:
            # If reading fails, skip table and downloads
            pass

st.markdown("---")
st.caption(
    "Developed by AlMSLab"
)


