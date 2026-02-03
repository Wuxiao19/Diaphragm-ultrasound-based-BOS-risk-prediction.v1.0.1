"""
FastMCP tool server wrapping `DetectionPipeline` for diaphragm ultrasound analysis.

Provides two tools (for the agent):
1. `detect_single_pair`: input one B-mode and one M-mode image path, return the risk probability.
2. `detect_batch_folders`: input a B-mode folder and an M-mode folder, batch infer risk probabilities.

Notes:
- Both tools reuse the logic and model weights from `integrated_detection_gui_ET.DetectionPipeline`.
- Filenames must follow the pipeline naming rule and include `YY-MM-DD-<ID>`, e.g. `24-05-01-C001_xxx.png`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Annotated

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from integrated_detection_gui_ET import DetectionPipeline


# ============================================================
# MCP server initialization
# ============================================================

mcp = FastMCP("Diaphragm-Ultrasound-ET-Detection")


# ============================================================
# Global pipeline management (avoid reloading models)
# ============================================================

_pipeline: Optional[DetectionPipeline] = None


def _get_pipeline() -> DetectionPipeline:
    """
    Get a DetectionPipeline instance with models loaded.

    - Use lazy loading + a global singleton to avoid reloading large models.
    - Disable GUI callback and log only to stdout.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline(gui_callback=None)
        _pipeline.load_models()
    return _pipeline


# ============================================================
# Pydantic return model definitions
# ============================================================


class SingleDetectionResult(BaseModel):
    """Detection result for a single B/M image pair."""

    b_image: str = Field(description="B-mode image filename used for inference (from pipeline output)")
    m_image: str = Field(description="M-mode image filename used for inference (from pipeline output)")
    merged_key: str = Field(
        description="Merged `merged_filename` (format YY-MM-DD-ID) that uniquely identifies an exam"
    )
    risk_probability: float = Field(
        description="Model risk probability (0-1, higher means higher risk)"
    )
    prediction: int = Field(
        description="Binary prediction label: 0=healthy (low risk), 1=diseased (high risk)"
    )
    prediction_label: str = Field(
        description="Prediction label text: 'healthy' or 'diseased'"
    )
    detect_output_dir: str = Field(
        description="Local output directory for this run (contains CSVs and other files)"
    )


class BatchDetectionItem(BaseModel):
    """Result for a single sample in batch detection."""

    b_image: str = Field(description="Merged B-mode filename(s) for this sample (semicolon-separated if multiple)")
    m_image: str = Field(description="Merged M-mode filename(s) for this sample (semicolon-separated if multiple)")
    merged_key: str = Field(description="Merged `merged_filename`, format YY-MM-DD-ID")
    risk_probability: float = Field(description="Risk probability for this sample")
    prediction: int = Field(description="Prediction label: 0=healthy, 1=diseased")
    prediction_label: str = Field(description="Prediction text label: 'healthy' or 'diseased'")


class BatchDetectionResult(BaseModel):
    """Batch detection result."""

    total_samples: int = Field(description="Number of successfully predicted samples (rows)")
    average_probability: float = Field(
        description="Average risk probability across samples (simple mean)"
    )
    items: List[BatchDetectionItem] = Field(
        description="Detailed prediction results for each sample"
    )
    detect_output_dir: str = Field(
        description="Output directory for this batch run (contains detect_result.csv, etc.)"
    )


"""
Internal implementation: single B/M image pair detection (shared by MCP tool and local testing).
"""
async def detect_single_pair_impl(
    b_image_path: str,
    m_image_path: str,
) -> SingleDetectionResult:
    """
    Path conversion: if the input path doesn't exist, try resolving it relative to
    the current working directory. This helps with cross-platform paths (Windows/Linux).
    """
    b_path = Path(b_image_path)
    m_path = Path(m_image_path)
    
    # If path does not exist, try relative or CWD-based resolution
    if not b_path.exists():
        current_dir = Path.cwd()
        filename = b_path.name
        possible_paths = []
        
        # 1. Original path (already checked, missing)
        possible_paths.append(b_path)
        
        # 2. If relative, try current working directory
        if not b_path.is_absolute():
            possible_paths.append(current_dir / b_path)
        
        # 3. If path contains "uploaded_inputs", try under current directory
        path_str = str(b_path)
        if "uploaded_inputs" in path_str:
            # Extract the portion after uploaded_inputs
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. Try using parent directory name + filename
        if len(b_path.parts) >= 2:
            parent_dir = b_path.parts[-2]
            possible_paths.append(current_dir / "uploaded_inputs" / parent_dir / filename)
        
        # 5. Search under uploaded_inputs in current directory (recursive)
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / filename
                    if candidate.exists():
                        possible_paths.append(candidate)
        
        # Deduplicate and check
        seen = set()
        unique_paths = []
        for p in possible_paths:
            p_str = str(p)
            if p_str not in seen:
                seen.add(p_str)
                unique_paths.append(p)
        
        found = False
        for p in unique_paths:
            if p.exists() and p.is_file():
                b_path = p
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"B-mode image not found: {b_image_path}\n"
                f"Tried paths: {[str(p) for p in unique_paths[:5]]}\n"
                f"Current working directory: {current_dir}\n"
                f"Please ensure the file is uploaded to the uploaded_inputs directory"
            )
    
    if not m_path.exists():
        # Same logic for M-mode
        current_dir = Path.cwd()
        filename = m_path.name
        possible_paths = []
        
        # 1. Original path (already checked, missing)
        possible_paths.append(m_path)
        
        # 2. If relative, try current working directory
        if not m_path.is_absolute():
            possible_paths.append(current_dir / m_path)
        
        # 3. If path contains "uploaded_inputs", try under current directory
        path_str = str(m_path)
        if "uploaded_inputs" in path_str:
            # Extract the portion after uploaded_inputs
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. Try using parent directory name + filename
        if len(m_path.parts) >= 2:
            parent_dir = m_path.parts[-2]
            possible_paths.append(current_dir / "uploaded_inputs" / parent_dir / filename)
        
        # 5. Search under uploaded_inputs in current directory (recursive)
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / filename
                    if candidate.exists():
                        possible_paths.append(candidate)
        
        # Deduplicate and check
        seen = set()
        unique_paths = []
        for p in possible_paths:
            p_str = str(p)
            if p_str not in seen:
                seen.add(p_str)
                unique_paths.append(p)
        
        found = False
        for p in unique_paths:
            if p.exists() and p.is_file():
                m_path = p
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"M-mode image not found: {m_image_path}\n"
                f"Tried paths: {[str(p) for p in unique_paths[:5]]}\n"
                f"Current working directory: {current_dir}\n"
                f"Please ensure the file is uploaded to the uploaded_inputs directory"
            )

    pipeline = _get_pipeline()

    output_dir, results_df = pipeline.run(
        b_input=str(b_path),
        m_input=str(m_path),
        is_folder=False,
        use_csv=False,
    )

    if results_df is None or len(results_df) == 0:
        raise RuntimeError("The pipeline returned no results. Check if filenames follow the naming rule.")

    row = results_df.iloc[0]

    return SingleDetectionResult(
        b_image=str(row.get("b_filename", b_path.name)),
        m_image=str(row.get("m_filename", m_path.name)),
        merged_key=str(row.get("merged_filename", "")),
        risk_probability=float(row.get("risk_probability", 0.0)),
        prediction=int(row.get("prediction", 0)),
        prediction_label=str(row.get("prediction_label", "")),
        detect_output_dir=str(output_dir),
    )


# ============================================================
# Tool 1: single B/M image pair detection (MCP entrypoint)
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
    """
    Single-patient, single-exam inference interface (MCP tool wrapper).

    Internally calls `detect_single_pair_impl` to share the implementation across MCP and local scripts.
    """
    return await detect_single_pair_impl(b_image_path=b_image_path, m_image_path=m_image_path)


"""
Internal implementation: batch B/M folder detection (shared by MCP tool and local testing).
"""
async def detect_batch_folders_impl(
    b_folder_path: str,
    m_folder_path: str,
) -> BatchDetectionResult:
    """
    Path conversion: if the input path doesn't exist, try resolving it relative to
    the current working directory. This helps with cross-platform paths (Windows/Linux).
    """
    b_dir = Path(b_folder_path)
    m_dir = Path(m_folder_path)
    
    # If path does not exist, try relative or CWD-based resolution
    if not b_dir.exists() or not b_dir.is_dir():
        current_dir = Path.cwd()
        possible_paths = []
        
        # 1. Original path (already checked, missing)
        possible_paths.append(b_dir)
        
        # 2. If relative, try current working directory
        if not b_dir.is_absolute():
            possible_paths.append(current_dir / b_dir)
        
        # 3. If path contains "uploaded_inputs", try under current directory
        path_str = str(b_dir)
        if "uploaded_inputs" in path_str:
            # Extract the portion after uploaded_inputs
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. Try using directory name
        if b_dir.name:
            possible_paths.append(current_dir / "uploaded_inputs" / b_dir.name)
            possible_paths.append(current_dir / b_dir.name)
        
        # 5. Search under uploaded_inputs in current directory (recursive)
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir() and subdir.name == b_dir.name:
                    possible_paths.append(subdir)
        
        # Deduplicate and check
        seen = set()
        unique_paths = []
        for p in possible_paths:
            p_str = str(p)
            if p_str not in seen:
                seen.add(p_str)
                unique_paths.append(p)
        
        found = False
        for p in unique_paths:
            if p.exists() and p.is_dir():
                b_dir = p
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"B-mode image folder not found or not a folder: {b_folder_path}\n"
                f"Tried paths: {[str(p) for p in unique_paths[:5]]}\n"
                f"Current working directory: {current_dir}\n"
                f"Please ensure the folder is uploaded to the uploaded_inputs directory"
            )
    
    if not m_dir.exists() or not m_dir.is_dir():
        current_dir = Path.cwd()
        possible_paths = []
        
        # 1. Original path 
        possible_paths.append(m_dir)
        
        # 2. If relative, try current working directory
        if not m_dir.is_absolute():
            possible_paths.append(current_dir / m_dir)
        
        # 3. If path contains "uploaded_inputs", try under current directory
        path_str = str(m_dir)
        if "uploaded_inputs" in path_str:
            # Extract the portion after uploaded_inputs
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. Try using directory name
        if m_dir.name:
            possible_paths.append(current_dir / "uploaded_inputs" / m_dir.name)
            possible_paths.append(current_dir / m_dir.name)
        
        # 5. Search under uploaded_inputs in current directory (recursive)
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir() and subdir.name == m_dir.name:
                    possible_paths.append(subdir)
        
        # Deduplicate and check
        seen = set()
        unique_paths = []
        for p in possible_paths:
            p_str = str(p)
            if p_str not in seen:
                seen.add(p_str)
                unique_paths.append(p)
        
        found = False
        for p in unique_paths:
            if p.exists() and p.is_dir():
                m_dir = p
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"M-mode image folder not found or not a folder: {m_folder_path}\n"
                f"Tried paths: {[str(p) for p in unique_paths[:5]]}\n"
                f"Current working directory: {current_dir}\n"
                f"Please ensure the folder is uploaded to the uploaded_inputs directory"
            )

    pipeline = _get_pipeline()

    output_dir, results_df = pipeline.run(
        b_input=str(b_dir),
        m_input=str(m_dir),
        is_folder=True,
        use_csv=False,
    )

    if results_df is None or len(results_df) == 0:
        # Return empty result if no matches (missing_modality_samples.csv is still saved)
        return BatchDetectionResult(
            total_samples=0,
            average_probability=0.0,
            items=[],
            detect_output_dir=str(output_dir),
        )

    items: List[BatchDetectionItem] = []
    probs: List[float] = []

    for _, row in results_df.iterrows():
        prob = float(row.get("risk_probability", 0.0))
        probs.append(prob)
        items.append(
            BatchDetectionItem(
                b_image=str(row.get("b_filename", "")),
                m_image=str(row.get("m_filename", "")),
                merged_key=str(row.get("merged_filename", "")),
                risk_probability=prob,
                prediction=int(row.get("prediction", 0)),
                prediction_label=str(row.get("prediction_label", "")),
            )
        )

    avg_prob = float(np.mean(probs)) if probs else 0.0

    return BatchDetectionResult(
        total_samples=len(items),
        average_probability=avg_prob,
        items=items,
        detect_output_dir=str(output_dir),
    )


# ============================================================
# Tool 2: batch B/M folder detection (MCP entrypoint)
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
        str,
        Field(
            description=(
                "Folder path containing B-mode ultrasound images. Filenames must follow `YY-MM-DD-<ID>`."
            )
        ),
    ],
    m_folder_path: Annotated[
        str,
        Field(
            description=(
                "Folder path containing M-mode ultrasound images. Filenames must follow `YY-MM-DD-<ID>`."
            )
        ),
    ],
) -> BatchDetectionResult:
    """
    Batch inference interface for multiple patients/exams (MCP tool wrapper).

    Internally calls `detect_batch_folders_impl` to share the implementation across MCP and local scripts.
    """
    return await detect_batch_folders_impl(
        b_folder_path=b_folder_path,
        m_folder_path=m_folder_path,
    )


if __name__ == "__main__":
    mcp.run()


