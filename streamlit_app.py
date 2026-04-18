import os
from pathlib import Path
import shutil
import uuid
import re
import numpy as np
import pandas as pd
import streamlit as st
from llm_agent import run_llm_agent
import asyncio
import json

REFERENCE_FACTOR_COLUMNS = ["Sex", "Age", "BMI", "Complication", "cGVHD", "Time-HSCT"]
REFERENCE_IDENTIFIER_COLUMNS = ["merged_key", "patient_id", "date"]


# ============================================================
# Streamlit basic page configuration
# ============================================================

st.set_page_config(
    page_title="Diaphragm Ultrasound Analysis System",
    page_icon="🩺",
    layout="wide",
)

st.title(
    "Diaphragm Ultrasound Analysis System",
    help=(
        "Upload B-mode and M-mode diaphragm ultrasound images for one patient (single exam) "
        "or for multiple patients (batch exams).\n\n"
        "The system will automatically perform: feature extraction -> feature reduction "
        "-> feature fusion -> ExtraTrees-based binary classification."
    ),
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


def load_reference_table(uploaded_file) -> pd.DataFrame:
    """Load a CSV/XLSX reference table used only for LLM-side interpretation."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or XLSX.")


def normalize_reference_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded reference-factor columns without changing detection flow."""
    normalized = df.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]

    rename_map = {}
    for col in normalized.columns:
        lower_col = col.lower()
        if lower_col == "sex":
            rename_map[col] = "Sex"
        elif lower_col == "age":
            rename_map[col] = "Age"
        elif lower_col == "bmi":
            rename_map[col] = "BMI"
        elif lower_col == "complication":
            rename_map[col] = "Complication"
        elif lower_col == "cgvhd":
            rename_map[col] = "cGVHD"
        elif lower_col in {"time-hsct", "time_hsct", "time hsct"}:
            rename_map[col] = "Time-HSCT"
        elif lower_col == "merged_key":
            rename_map[col] = "merged_key"
        elif lower_col == "patient_id":
            rename_map[col] = "patient_id"
        elif lower_col == "date":
            rename_map[col] = "date"

    normalized = normalized.rename(columns=rename_map)
    if "date" in normalized.columns:
        normalized["date"] = normalized["date"].apply(normalize_reference_date_value)
    if "patient_id" in normalized.columns:
        normalized["patient_id"] = normalized["patient_id"].apply(
            lambda v: str(v).strip() if pd.notna(v) else None
        )
    if "merged_key" in normalized.columns:
        normalized["merged_key"] = normalized["merged_key"].apply(
            lambda v: str(v).strip() if pd.notna(v) else None
        )
    elif {"patient_id", "date"}.issubset(normalized.columns):
        normalized["merged_key"] = normalized.apply(
            lambda row: (
                f"{row['date']}-{row['patient_id']}"
                if row.get("date") and row.get("patient_id")
                else None
            ),
            axis=1,
        )
    return normalized


def normalize_reference_date_value(value):
    """Normalize reference-table dates to YY-MM-DD for matching and prompt use."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%y-%m-%d")

    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{2}-\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}:\d{2})?", text):
        return pd.to_datetime(text).strftime("%y-%m-%d")
    return text


def serialize_reference_records(df: pd.DataFrame) -> list[dict]:
    """Convert reference DataFrame to JSON-safe records for agent context."""
    records = []
    for row in df.to_dict(orient="records"):
        clean_row = {}
        for key, value in row.items():
            if pd.isna(value):
                clean_row[key] = None
            elif key == "date":
                clean_row[key] = normalize_reference_date_value(value)
            elif isinstance(value, np.generic):
                clean_row[key] = value.item()
            else:
                clean_row[key] = value
        records.append(clean_row)
    return records


def build_batch_reference_preview(reference_df: pd.DataFrame, detection_summary: dict | None = None) -> pd.DataFrame:
    """Prepare a UI preview that aligns uploaded reference factors with prediction results when possible."""
    if reference_df is None or reference_df.empty:
        return pd.DataFrame()

    preview_df = reference_df.copy()
    preferred_cols = [c for c in REFERENCE_IDENTIFIER_COLUMNS + REFERENCE_FACTOR_COLUMNS if c in preview_df.columns]
    remaining_cols = [c for c in preview_df.columns if c not in preferred_cols]
    preview_df = preview_df[preferred_cols + remaining_cols]

    if not detection_summary:
        return preview_df

    items = detection_summary.get("items", [])
    if not items:
        return preview_df

    items_df = pd.DataFrame(items)
    if items_df.empty:
        return preview_df

    if "merged_key" not in preview_df.columns:
        if {"patient_id", "date"}.issubset(preview_df.columns):
            preview_df["merged_key"] = (
                preview_df["date"].astype(str) + "-" + preview_df["patient_id"].astype(str)
            )
        else:
            return preview_df

    if "merged_key" not in items_df.columns:
        return preview_df

    merged = items_df[["merged_key", "risk_probability", "prediction_label"]].merge(
        preview_df, on="merged_key", how="left"
    )
    cols = ["merged_key", "risk_probability", "prediction_label"]
    cols += [c for c in REFERENCE_IDENTIFIER_COLUMNS if c in merged.columns and c != "merged_key"]
    cols += [c for c in REFERENCE_FACTOR_COLUMNS if c in merged.columns]
    cols += [c for c in merged.columns if c not in cols]
    return merged[cols]


def parse_optional_number(value: str, cast_type):
    """Convert optional text input to number for LLM-only reference fields."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return cast_type(raw)
    except Exception:
        return None


def get_reference_row_for_case(reference_context: dict | None, row: dict | None) -> dict:
    """Match optional reference factors to a single detection row for display only."""
    if not isinstance(reference_context, dict) or not isinstance(row, dict):
        return {}

    mode = reference_context.get("mode")
    if mode == "single":
        values = reference_context.get("single_values") or {}
        return {k: v for k, v in values.items() if v not in (None, "", [])}

    if mode != "folder":
        return {}

    records = reference_context.get("batch_df") or []
    merged_key = str(row.get("merged_key") or "").strip()
    patient_id = str(row.get("patient_id") or "").strip()
    date = normalize_reference_date_value(row.get("date"))

    for record in records:
        if not isinstance(record, dict):
            continue
        record_key = str(record.get("merged_key") or "").strip()
        record_pid = str(record.get("patient_id") or "").strip()
        record_date = normalize_reference_date_value(record.get("date"))

        if merged_key and record_key and merged_key == record_key:
            return {k: record.get(k) for k in REFERENCE_FACTOR_COLUMNS if record.get(k) not in (None, "", [])}
        if patient_id and date and record_pid == patient_id and record_date == date:
            return {k: record.get(k) for k in REFERENCE_FACTOR_COLUMNS if record.get(k) not in (None, "", [])}

    return {}

# Initialize session_state
if "detect_output_dir" not in st.session_state:
    st.session_state["detect_output_dir"] = None


def _on_file_uploader_change(mode: str) -> None:
    """Clear previous detection results when the user uploads new files."""
    st.session_state.pop("agent_result", None)
    st.session_state["detect_output_dir"] = None


def _render_agent_result(ar: dict) -> None:
    """Render agent result stored in session_state or returned from run_llm_agent.

    This is separated so the UI remains visible across Streamlit reruns 
    """
    if not ar:
        return

    with st.expander("🔧 View: Tools called by LLM", expanded=False):
        if ar.get("tool_calls"):
            for i, tc in enumerate(ar["tool_calls"], 1):
                st.write(f"**Tool {i}**: `{tc['name']}`")
        else:
            st.write("(No tools called this run)")
    
    st.markdown("### 💬 LLM Agent full analysis")
    detection_summary = None
    if isinstance(ar.get("tool_results"), dict):
        for _, res in ar.get("tool_results", {}).items():
            if isinstance(res, dict) and res:
                summary = res.get("detection_summary")
                if isinstance(summary, dict):
                    detection_summary = summary
                    break

    reference_context = st.session_state.get("reference_context")

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

        st.markdown("**Sample details**")
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

            styled = display_df.style.format({"risk_probability": "{:.3f}"}).map(
                _risk_color, subset=["risk_probability"]
            )
            st.dataframe(styled, use_container_width=True)

        high_risk = items_df[items_df["risk_probability"] > 0.6] if not items_df.empty else pd.DataFrame()
        if not high_risk.empty:
            with st.expander("High-risk patient images (B/M mode)", expanded=False):
                for _, row in high_risk.iterrows():
                    b_name = row.get("b_filename") or ""
                    m_name = row.get("m_filename") or ""
                    b_path = _find_image_path(str(b_name)) if b_name else None
                    m_path = _find_image_path(str(m_name)) if m_name else None

                    st.markdown(
                        f"- Patient {row.get('patient_id')} | {row.get('date')} | risk={float(row.get('risk_probability', 0.0)):.3f}"
                    )
                    matched_reference = get_reference_row_for_case(reference_context, row.to_dict())
                    if matched_reference:
                        st.caption(
                            "Clinical reference factors used for interpretation: "
                            + ", ".join(f"{k}={v}" for k, v in matched_reference.items())
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
                    visits_df = pd.DataFrame(rp.get("visits", []))
                    if not visits_df.empty and isinstance(reference_context, dict) and reference_context:
                        visits_df = visits_df.copy()
                        visits_df["Clinical reference factors"] = visits_df.apply(
                            lambda x: ", ".join(
                                f"{k}: {v}" for k, v in get_reference_row_for_case(reference_context, x.to_dict()).items()
                            ) or "-",
                            axis=1,
                        )
                    st.dataframe(visits_df, use_container_width=True)

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

st.subheader(
    "1. Upload input data",
    help=(
        "File naming rule: each filename must start with `YY-MM-DD-<ID>`, "
        "e.g. `24-05-01-A001_xxx.png`. The same patient ID on the same date "
        "will be merged as one exam."
    ),
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
# Reference factors for LLM-only interpretation
# ============================================================
st.markdown("---")
st.subheader(
    "2. Clinical reference factors for LLM interpretation",
    help=(
        "These factors are optional and are used only as reference information in the LLM suggestion stage. "
        "They do not change image processing, feature extraction, or risk prediction."
    ),
)

reference_context = None

if input_mode == "single":
    ref_col1, ref_col2, ref_col3 = st.columns(3)
    with ref_col1:
        ref_sex = st.selectbox("Sex", options=["", "Male", "Female"], index=0)
        ref_age_text = st.text_input("Age")
    with ref_col2:
        ref_bmi_text = st.text_input("BMI")
        ref_complication = st.text_input("Complication")
    with ref_col3:
        ref_cgvhd = st.text_input("cGVHD")
        ref_time_hsct = st.text_input("Time-HSCT")

    single_reference_values = {
        "Sex": ref_sex or None,
        "Age": parse_optional_number(ref_age_text, int),
        "BMI": parse_optional_number(ref_bmi_text, float),
        "Complication": ref_complication.strip() or None,
        "cGVHD": ref_cgvhd.strip() or None,
        "Time-HSCT": ref_time_hsct.strip() or None,
    }
    single_reference_values = {k: v for k, v in single_reference_values.items() if v not in (None, "")}
    reference_context = {"mode": "single", "single_values": single_reference_values}
else:
    batch_reference_file = st.file_uploader(
        "Upload batch reference table (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        key="batch_reference_file",
        help=(
            "Supported format: CSV/XLSX with one row per case. Use `merged_key` to match directly, "
            "or provide both `patient_id` and `date`. Optional reference columns: `Sex`, `Age`, `BMI`, "
            "`Complication`, `cGVHD`, `Time-HSCT`. Example: merged_key=24-05-01-A001 or patient_id=A001, date=24-05-01."
        ),
    )
    batch_reference_df = pd.DataFrame()
    if batch_reference_file is not None:
        try:
            batch_reference_df = normalize_reference_df(load_reference_table(batch_reference_file))
            with st.expander("Uploaded batch reference preview", expanded=False):
                st.dataframe(batch_reference_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read reference table: {e}")
            batch_reference_df = pd.DataFrame()
    reference_context = {
        "mode": "folder",
        "batch_df": serialize_reference_records(batch_reference_df) if not batch_reference_df.empty else [],
        "source_filename": getattr(batch_reference_file, "name", None) if batch_reference_file else None,
    }

# ============================================================
# Global: LLM Agent mode
# ============================================================
st.markdown("---")
st.subheader(
    "3. AI detection and interpretation",
    help=(
        "Note: In this mode, simply upload B-mode and M-mode diaphragm ultrasound images above. "
        "Large Language Model will call the backend detection pipeline (MCP tools), complete feature extraction "
        "and risk prediction, then generate an English interpretation. This cannot replace a doctor's diagnosis."
    ),
)

llm_secret_key = ""
try:
    llm_secret_key = st.secrets.get("llm_api_key", "")
except Exception:
    llm_secret_key = ""

st.info("🤖 **Agent mode**: directly use your uploaded images, call backend detection tools, and generate a full analysis.")

if st.button("🚀 Run LLM Agent", type="primary"):
    final_llm_key = (llm_secret_key or "").strip()
    if not final_llm_key:
        st.error("Please set llm_api_key in secrets.toml.")
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
            st.session_state["reference_context"] = reference_context or {}

            with st.spinner("🤖 LLM Agent is working: calling detection tools and generating analysis..."):
                if b_path_for_agent and m_path_for_agent:
                    agent_result = asyncio.run(run_llm_agent(
                        b_image_path=b_path_for_agent,
                        m_image_path=m_path_for_agent,
                        api_key=final_llm_key,
                        single_reference_factors=(reference_context or {}).get("single_values"),
                    ))
                elif b_folder_for_agent and m_folder_for_agent:
                    agent_result = asyncio.run(run_llm_agent(
                        b_folder_path=b_folder_for_agent,
                        m_folder_path=m_folder_for_agent,
                        api_key=final_llm_key,
                        batch_reference_records=(reference_context or {}).get("batch_df"),
                        batch_reference_filename=(reference_context or {}).get("source_filename"),
                    ))
                else:
                    raise ValueError("Unable to determine single or batch mode. Please check your uploads.")

            st.success("✅ LLM Agent analysis complete!")

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
            st.error(f"❌ LLM Agent failed: {e}")
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
# Global: show CSV preview and download 
# ====================================================
detect_output_dir = st.session_state.get("detect_output_dir")
if isinstance(detect_output_dir, str) and detect_output_dir:
    result_csv_path = os.path.join(detect_output_dir, "detect_result.csv")
    if os.path.exists(result_csv_path):
        try:
            results_df = pd.read_csv(result_csv_path)

            st.markdown("---")
            st.markdown("### 📥 Download CSV Results")

            # Detection results subsection
            with st.expander("📊 Detection Results", expanded=False):
                key_cols = ["merged_key","b_filename","m_filename",
                    "risk_probability","prediction","prediction_label",]
                show_cols = [c for c in key_cols if c in results_df.columns]
                st.dataframe(results_df[show_cols] if show_cols else results_df, use_container_width=True)

                st.download_button(
                    label="📥 Download Detection Results CSV",
                    data=results_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="detect_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Missing modality subsection
            missing_csv_path = os.path.join(detect_output_dir, "missing_modality_samples.csv")
            if os.path.exists(missing_csv_path):
                try:
                    missing_df = pd.read_csv(missing_csv_path)
                except Exception:
                    missing_df = None

                if missing_df is not None and not missing_df.empty:
                    with st.expander("⚠️ Missing Modality Samples", expanded=False):
                        st.caption(
                            f"Found {len(missing_df)} samples with incomplete B/M pairs (not included in prediction)"
                        )
                        st.dataframe(missing_df, use_container_width=True)

                        st.download_button(
                            label="📥 Download Missing Modality CSV",
                            data=missing_df.to_csv(index=False, encoding="utf-8-sig"),
                            file_name="missing_modality_samples.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
        except Exception:
            # If reading fails, skip table and downloads
            pass

st.markdown("---")
st.caption("Developed by AlMSLab")


