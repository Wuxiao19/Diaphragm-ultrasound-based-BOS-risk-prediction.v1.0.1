"""
FastMCP tool server wrapping `DetectionPipeline` for diaphragm ultrasound analysis.

two tools (for the agent):
1. `detect_single_pair`: input one B-mode and one M-mode image path, return the risk probability.
2. `detect_batch_folders`: input a B-mode folder and an M-mode folder, batch infer risk probabilities.

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Annotated, Dict, Any
import numpy as np
import pandas as pd
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from detection_pipeline import DetectionPipeline, parse_date_and_patient_id

# ============================================================
# MCP server initialization
# ============================================================

mcp = FastMCP("Diaphragm-Ultrasound-ET-Detection")

# ============================================================
# Global pipeline management 
# ============================================================

_pipeline: Optional[DetectionPipeline] = None

def _get_pipeline() -> DetectionPipeline:
    """
    Get a DetectionPipeline instance with models loaded.

    """
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline(gui_callback=None)
        _pipeline.load_models()
    return _pipeline

# ============================================================
# Path resolution utilities 
# ============================================================

def _resolve_path(p: Path, is_dir: bool = False, which: str = None) -> Path:
    """
    Resolve a file or directory path by trying several fallbacks.

    Args:
        p: Path to resolve
        is_dir: True if expecting a directory, False for file
        which: Description for error message (e.g., "B-mode image")

    Raises:
        FileNotFoundError: If no valid path is found
    """
    if which is None:
        which = "folder" if is_dir else "file"

    current_dir = Path.cwd()
    possible_paths = []

    # 1. Original path
    possible_paths.append(p)

    # 2. If relative, try current working directory
    if not p.is_absolute():
        possible_paths.append(current_dir / p)

    # 3. If path contains "uploaded_inputs", try under current directory
    path_str = str(p)
    if "uploaded_inputs" in path_str:
        parts = path_str.split("uploaded_inputs", 1)
        if len(parts) > 1:
            rel_part = parts[1].lstrip("/\\")
            possible_paths.append(current_dir / "uploaded_inputs" / rel_part)

    # 4. Try using name under uploaded_inputs
    if p.name:
        possible_paths.append(current_dir / "uploaded_inputs" / p.name)
        if not is_dir and len(p.parts) >= 2:
            # For files, also try parent_dir/filename
            parent_dir = p.parts[-2]
            possible_paths.append(current_dir / "uploaded_inputs" / parent_dir / p.name)

    # 5. Search under uploaded_inputs recursively
    uploaded_dir = current_dir / "uploaded_inputs"
    if uploaded_dir.exists():
        for subdir in uploaded_dir.iterdir():
            if subdir.is_dir():
                if is_dir and subdir.name == p.name:
                    possible_paths.append(subdir)
                elif not is_dir:
                    candidate = subdir / p.name
                    if candidate.exists():
                        possible_paths.append(candidate)

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for candidate in possible_paths:
        s = str(candidate)
        if s not in seen:
            seen.add(s)
            unique_paths.append(candidate)

    # Find first valid path
    for candidate in unique_paths:
        if is_dir:
            if candidate.exists() and candidate.is_dir():
                return candidate
        else:
            if candidate.exists() and candidate.is_file():
                return candidate

    # Not found
    type_desc = "folder" if is_dir else "file"
    raise FileNotFoundError(
        f"{which.capitalize()} not found: {p}\n"
        f"Tried paths: {[str(x) for x in unique_paths[:10]]}\n"
        f"Current working directory: {current_dir}\n"
        f"Please ensure the {type_desc} is uploaded to the uploaded_inputs directory"
    )


# ============================================================
# Pydantic return model definitions
# ============================================================

class SingleDetectionResult(BaseModel):
    """Detection result for a single B/M image pair."""

    b_filename: str = Field(description="B-mode image filename used for inference (from pipeline output)")
    m_filename: str = Field(description="M-mode image filename used for inference (from pipeline output)")
    merged_key: str = Field(description="Merged `merged_key` (format YY-MM-DD-ID) that uniquely identifies an exam")
    risk_probability: float = Field(description="Model risk probability (0-1, higher means higher risk)")
    prediction: int = Field(description="Binary prediction label: 0=healthy (low risk), 1=diseased (high risk)")
    prediction_label: str = Field(description="Prediction label text: 'healthy' or 'diseased'")
    detect_output_dir: str = Field(description="Local output directory for this run (contains CSVs and other files)")
    detection_summary: Dict[str, Any] = Field(description="Pre-built summary for agent/LLM consumption")


class BatchDetectionItem(BaseModel):
    """Result for a single sample in batch detection."""

    b_filename: str = Field(description="Merged B-mode filename(s) for this sample (semicolon-separated if multiple)")
    m_filename: str = Field(description="Merged M-mode filename(s) for this sample (semicolon-separated if multiple)")
    merged_key: str = Field(description="Merged `merged_key`, format YY-MM-DD-ID")
    risk_probability: float = Field(description="Risk probability for this sample")
    prediction: int = Field(description="Prediction label: 0=healthy, 1=diseased")
    prediction_label: str = Field(description="Prediction text label: 'healthy' or 'diseased'")


class BatchDetectionResult(BaseModel):
    """Batch detection result."""

    total_samples: int = Field(description="Number of successfully predicted samples (rows)")
    items: List[BatchDetectionItem] = Field(description="Detailed prediction results for each sample")
    detect_output_dir: str = Field(description="Output directory for this batch run (contains detect_result.csv, etc.)")
    detection_summary: Dict[str, Any] = Field(description="Pre-built summary for agent/LLM consumption")


def _build_missing_modality_summary(detect_output_dir: str) -> Dict[str, Any] | None:
    if not detect_output_dir:
        return None
    missing_csv_path = os.path.join(detect_output_dir, "missing_modality_samples.csv")
    if not os.path.exists(missing_csv_path):
        return None
    try:
        df_missing = pd.read_csv(missing_csv_path)
        total_missing = int(len(df_missing))
        by_type = (
            df_missing["missing_modality"].value_counts().to_dict()
            if "missing_modality" in df_missing.columns
            else {}
        )
        by_patient: Dict[str, int] = {}
        if "patient_id" in df_missing.columns:
            by_patient = df_missing["patient_id"].value_counts().to_dict()
        return {
            "total_missing_samples": total_missing,
            "missing_by_type": by_type,
            "missing_by_patient": by_patient,
            "csv_path": missing_csv_path,
        }
    except Exception:
        return None

def _build_detection_summary(mode: str,items: List[Dict[str, Any]],detect_output_dir: str,) -> Dict[str, Any]:
    visits_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        pid = it.get("patient_id")
        if not pid:
            continue
        visits_by_pid.setdefault(pid, []).append(it)

    recheck_patients: List[Dict[str, Any]] = []
    for pid, visits in visits_by_pid.items():
        dates = sorted({v.get("date") for v in visits if v.get("date")})
        if len(dates) > 1:
            recheck_patients.append(
                {
                    "patient_id": pid,
                    "exam_dates": dates,
                    "visits": sorted(visits, key=lambda x: (str(x.get("date") or ""), x.get("merged_key", ""))
                    ),
                }
            )

    summary: Dict[str, Any] = {
        "mode": mode,
        "total_samples": len(items),
        "items": items,
        "recheck_patients": recheck_patients,
    }

    missing_summary = _build_missing_modality_summary(detect_output_dir)
    if missing_summary is not None:
        summary["missing_modality_summary"] = missing_summary

    return summary

# ============================================================
# Tool 1: single B/M image pair detection
# ============================================================

@mcp.tool(
    name="detect_single_pair",
    description=(
        "Use the ExtraTrees pipeline to detect risk probability for a B-mode and M-mode diaphragm ultrasound pair."
        "Requirement: filenames must contain `YY-MM-DD-<ID>`, e.g. `24-05-01-C001_xxx.png`."
    ),
)
async def detect_single_pair(
    b_image_path: Annotated[
        str, Field(description="Absolute or relative path to a single B-mode ultrasound image")
    ],
    m_image_path: Annotated[
        str, Field(description="Absolute or relative path to a single M-mode ultrasound image")
    ],
) -> SingleDetectionResult:
    """Single-patient, single-exam inference."""

    b_path = Path(b_image_path)
    m_path = Path(m_image_path)

    if not b_path.exists():
        b_path = _resolve_path(b_path, is_dir=False, which="B-mode image")
    if not m_path.exists():
        m_path = _resolve_path(m_path, is_dir=False, which="M-mode image")

    pipeline = _get_pipeline()

    output_dir, results_df = pipeline.run(
        b_input=str(b_path),
        m_input=str(m_path),
        is_folder=False,
    )

    if results_df is None or len(results_df) == 0:
        raise RuntimeError("The pipeline returned no results. Check if filenames follow the naming rule.")

    row = results_df.iloc[0]

    merged_key = str(row.get("merged_key", ""))
    date_str, pid = parse_date_and_patient_id(merged_key)
    summary_items = [
        {
            "merged_key": merged_key,
            "date": date_str,
            "patient_id": pid,
            "risk_probability": float(row.get("risk_probability", 0.0)),
            "prediction": int(row.get("prediction", 0)),
            "prediction_label": str(row.get("prediction_label", "")),
            "b_filename": str(row.get("b_filename", b_path.name)),
            "m_filename": str(row.get("m_filename", m_path.name)),
        }
    ]

    detection_summary = _build_detection_summary("single", summary_items, str(output_dir))

    return SingleDetectionResult(
        b_filename=str(row.get("b_filename", b_path.name)),
        m_filename=str(row.get("m_filename", m_path.name)),
        merged_key=merged_key,
        risk_probability=float(row.get("risk_probability", 0.0)),
        prediction=int(row.get("prediction", 0)),
        prediction_label=str(row.get("prediction_label", "")),
        detect_output_dir=str(output_dir),
        detection_summary=detection_summary,
    )


# ============================================================
# Tool 2: batch B/M folder detection
# ============================================================

@mcp.tool(
    name="detect_batch_folders",
    description=(
        "Batch detection: input a B-mode image folder and an M-mode image folder."
        "The pipeline pairs and merges by `YY-MM-DD-<ID>` and returns risk probabilities for each sample."
    ),
)
async def detect_batch_folders(
    b_folder_path: Annotated[
        str, Field(description="Folder path containing B-mode ultrasound images. Filenames must follow `YY-MM-DD-<ID>`.")
    ],
    m_folder_path: Annotated[
        str, Field(description="Folder path containing M-mode ultrasound images. Filenames must follow `YY-MM-DD-<ID>`.")
    ],
) -> BatchDetectionResult:
    """Batch inference for multiple patients/exams."""

    b_dir = Path(b_folder_path)
    m_dir = Path(m_folder_path)

    if not b_dir.exists() or not b_dir.is_dir():
        b_dir = _resolve_path(b_dir, is_dir=True, which="B-mode image folder")
    if not m_dir.exists() or not m_dir.is_dir():
        m_dir = _resolve_path(m_dir, is_dir=True, which="M-mode image folder")

    pipeline = _get_pipeline()

    output_dir, results_df = pipeline.run(
        b_input=str(b_dir),
        m_input=str(m_dir),
        is_folder=True,
    )

    if results_df is None or len(results_df) == 0:
        detection_summary = _build_detection_summary("batch", [], str(output_dir))
        return BatchDetectionResult(
            total_samples=0,
            items=[],
            detect_output_dir=str(output_dir),
            detection_summary=detection_summary,
        )

    items: List[BatchDetectionItem] = []
    summary_items: List[Dict[str, Any]] = []
    for _, row in results_df.iterrows():
        merged_key = str(row.get("merged_key", ""))
        date_str, pid = parse_date_and_patient_id(merged_key)
        prob = float(row.get("risk_probability", 0.0))
        items.append(
            BatchDetectionItem(
                b_filename=str(row.get("b_filename", "")),
                m_filename=str(row.get("m_filename", "")),
                merged_key=merged_key,
                risk_probability=prob,
                prediction=int(row.get("prediction", 0)),
                prediction_label=str(row.get("prediction_label", "")),
            )
        )
        summary_items.append(
            {
                "merged_key": merged_key,
                "date": date_str,
                "patient_id": pid,
                "risk_probability": prob,
                "prediction": int(row.get("prediction", 0)),
                "prediction_label": str(row.get("prediction_label", "")),
                "b_filename": str(row.get("b_filename", "")),
                "m_filename": str(row.get("m_filename", "")),
            }
        )

    detection_summary = _build_detection_summary("batch", summary_items, str(output_dir))

    return BatchDetectionResult(
        total_samples=len(items),
        items=items,
        detect_output_dir=str(output_dir),
        detection_summary=detection_summary,
    )


if __name__ == "__main__":
    mcp.run()


