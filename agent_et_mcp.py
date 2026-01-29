"""
FastMCP tool server wrapping `DetectionPipeline` for diaphragm ultrasound analysis.

提供两个工具（给 Agent 使用）：
1. `detect_single_pair`：输入一张 B 超和一张 M 超图片路径，返回该检查的患病概率。
2. `detect_batch_folders`：输入一个 B 超图片文件夹和一个 M 超图片文件夹，批量推理多名患者的患病概率。

注意：
- 这两个工具完全复用 `integrated_detection_gui_ET.DetectionPipeline` 的逻辑和模型权重。
- 文件命名规则需满足原始流水线要求：文件名中包含 `YY-MM-DD-<ID>`，例如 `24-05-01-C001_xxx.png`。
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
# MCP Server 初始化
# ============================================================

mcp = FastMCP("Diaphragm-Ultrasound-ET-Detection")


# ============================================================
# 全局 Pipeline 管理（避免重复加载模型）
# ============================================================

_pipeline: Optional[DetectionPipeline] = None


def _get_pipeline() -> DetectionPipeline:
    """
    获取一个已经加载好模型的 DetectionPipeline 实例。

    - 采用懒加载 + 全局单例，避免每次调用都重新加载大模型。
    - 不使用 GUI 回调，日志只打印到 stdout。
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline(gui_callback=None)
        _pipeline.load_models()
    return _pipeline


# ============================================================
# Pydantic 返回结果模型定义
# ============================================================


class SingleDetectionResult(BaseModel):
    """单组 B/M 图片检测结果"""

    b_image: str = Field(description="参与推理的 B 超图片文件名（来自流水线结果）")
    m_image: str = Field(description="参与推理的 M 超图片文件名（来自流水线结果）")
    merged_key: str = Field(
        description="合并后的 `merged_filename`（形如 YY-MM-DD-ID，用于唯一标识一次检查）"
    )
    risk_probability: float = Field(
        description="模型输出的患病概率（0~1，越大表示越可能为高风险）"
    )
    prediction: int = Field(
        description="二分类预测结果标签：0=healthy（低风险），1=diseased（高风险）"
    )
    prediction_label: str = Field(
        description="预测结果文字标签：'healthy' 或 'diseased'"
    )
    detect_output_dir: str = Field(
        description="本次推理流水线在本地保存结果的目录路径（包含 CSV 等中间文件）"
    )


class BatchDetectionItem(BaseModel):
    """批量检测时单个样本的结果"""

    b_image: str = Field(description="该样本对应的（合并后）B 模态文件名（可能为多张，用分号分隔）")
    m_image: str = Field(description="该样本对应的（合并后）M 模态文件名（可能为多张，用分号分隔）")
    merged_key: str = Field(description="合并后的 `merged_filename`，形如 YY-MM-DD-ID")
    risk_probability: float = Field(description="该样本的患病概率")
    prediction: int = Field(description="预测标签：0=healthy，1=diseased")
    prediction_label: str = Field(description="预测文字标签：'healthy' 或 'diseased'")


class BatchDetectionResult(BaseModel):
    """批量检测返回结果"""

    total_samples: int = Field(description="成功完成预测的样本数量（即结果行数）")
    average_probability: float = Field(
        description="所有样本患病概率的平均值（简单平均）"
    )
    items: List[BatchDetectionItem] = Field(
        description="每个样本的详细预测结果列表"
    )
    detect_output_dir: str = Field(
        description="本次批量推理保存结果的目录路径（包含 detect_result.csv 等文件）"
    )


"""
内部实现函数：单组 B/M 图片检测（供 MCP 工具和本地测试脚本共同使用）
"""
async def detect_single_pair_impl(
    b_image_path: str,
    m_image_path: str,
) -> SingleDetectionResult:
    """
    处理路径转换：如果传入的路径不存在，尝试将其转换为相对于当前工作目录的路径。
    这样可以处理跨平台（Windows/Linux）的路径问题。
    """
    b_path = Path(b_image_path)
    m_path = Path(m_image_path)
    
    # 如果路径不存在，尝试转换为相对路径或基于当前工作目录的路径
    if not b_path.exists():
        current_dir = Path.cwd()
        filename = b_path.name
        possible_paths = []
        
        # 1. 原始路径（已检查，不存在）
        possible_paths.append(b_path)
        
        # 2. 如果是相对路径，尝试相对于当前工作目录
        if not b_path.is_absolute():
            possible_paths.append(current_dir / b_path)
        
        # 3. 如果路径包含 "uploaded_inputs"，尝试在当前目录下查找
        path_str = str(b_path)
        if "uploaded_inputs" in path_str:
            # 提取 uploaded_inputs 之后的部分
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. 尝试从路径中提取父目录名和文件名
        if len(b_path.parts) >= 2:
            parent_dir = b_path.parts[-2]
            possible_paths.append(current_dir / "uploaded_inputs" / parent_dir / filename)
        
        # 5. 尝试直接在当前目录的 uploaded_inputs 下查找（递归搜索）
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / filename
                    if candidate.exists():
                        possible_paths.append(candidate)
        
        # 去重并检查
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
                f"B 模式图像不存在: {b_image_path}\n"
                f"已尝试的路径: {[str(p) for p in unique_paths[:5]]}\n"
                f"当前工作目录: {current_dir}\n"
                f"请确保文件已上传到 uploaded_inputs 目录"
            )
    
    if not m_path.exists():
        # 同样的处理逻辑
        current_dir = Path.cwd()
        filename = m_path.name
        possible_paths = []
        
        # 1. 原始路径（已检查，不存在）
        possible_paths.append(m_path)
        
        # 2. 如果是相对路径，尝试相对于当前工作目录
        if not m_path.is_absolute():
            possible_paths.append(current_dir / m_path)
        
        # 3. 如果路径包含 "uploaded_inputs"，尝试在当前目录下查找
        path_str = str(m_path)
        if "uploaded_inputs" in path_str:
            # 提取 uploaded_inputs 之后的部分
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. 尝试从路径中提取父目录名和文件名
        if len(m_path.parts) >= 2:
            parent_dir = m_path.parts[-2]
            possible_paths.append(current_dir / "uploaded_inputs" / parent_dir / filename)
        
        # 5. 尝试直接在当前目录的 uploaded_inputs 下查找（递归搜索）
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / filename
                    if candidate.exists():
                        possible_paths.append(candidate)
        
        # 去重并检查
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
                f"M 模式图像不存在: {m_image_path}\n"
                f"已尝试的路径: {[str(p) for p in unique_paths[:5]]}\n"
                f"当前工作目录: {current_dir}\n"
                f"请确保文件已上传到 uploaded_inputs 目录"
            )

    pipeline = _get_pipeline()

    output_dir, results_df = pipeline.run(
        b_input=str(b_path),
        m_input=str(m_path),
        is_folder=False,
        use_csv=False,
    )

    if results_df is None or len(results_df) == 0:
        raise RuntimeError("流水线未返回任何结果，请检查图像文件名是否满足命名规则。")

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
# 工具 1：单组 B/M 图片检测（MCP 对外暴露的入口）
# ============================================================


@mcp.tool(
    name="detect_single_pair",
    description=(
        "使用 ExtraTrees 流水线检测一组 B 模式和 M 模式膈肌超声图像的患病概率。"
        "要求：文件名中包含 `YY-MM-DD-<ID>` 模式，例如 `24-05-01-C001_xxx.png`。"
    ),
)
async def detect_single_pair(
    b_image_path: Annotated[
        str, Field(description="B 模式超声图像的绝对路径或相对路径（单张图片文件）")
    ],
    m_image_path: Annotated[
        str, Field(description="M 模式超声图像的绝对路径或相对路径（单张图片文件）")
    ],
) -> SingleDetectionResult:
    """
    单患者单次检查（1 张 B 图 + 1 张 M 图）的推理接口（MCP 工具包装）。

    内部调用 `detect_single_pair_impl`，方便本地脚本和 MCP 共用同一套实现。
    """
    return await detect_single_pair_impl(b_image_path=b_image_path, m_image_path=m_image_path)


"""
内部实现函数：B/M 文件夹批量检测（供 MCP 工具和本地测试脚本共同使用）
"""
async def detect_batch_folders_impl(
    b_folder_path: str,
    m_folder_path: str,
) -> BatchDetectionResult:
    """
    处理路径转换：如果传入的路径不存在，尝试将其转换为相对于当前工作目录的路径。
    这样可以处理跨平台（Windows/Linux）的路径问题。
    """
    b_dir = Path(b_folder_path)
    m_dir = Path(m_folder_path)
    
    # 如果路径不存在，尝试转换为相对路径或基于当前工作目录的路径
    if not b_dir.exists() or not b_dir.is_dir():
        current_dir = Path.cwd()
        possible_paths = []
        
        # 1. 原始路径（已检查，不存在）
        possible_paths.append(b_dir)
        
        # 2. 如果是相对路径，尝试相对于当前工作目录
        if not b_dir.is_absolute():
            possible_paths.append(current_dir / b_dir)
        
        # 3. 如果路径包含 "uploaded_inputs"，尝试在当前目录下查找
        path_str = str(b_dir)
        if "uploaded_inputs" in path_str:
            # 提取 uploaded_inputs 之后的部分
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. 尝试从路径中提取目录名
        if b_dir.name:
            possible_paths.append(current_dir / "uploaded_inputs" / b_dir.name)
            possible_paths.append(current_dir / b_dir.name)
        
        # 5. 尝试直接在当前目录的 uploaded_inputs 下查找（递归搜索）
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir() and subdir.name == b_dir.name:
                    possible_paths.append(subdir)
        
        # 去重并检查
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
                f"B 模式图像文件夹不存在或不是文件夹: {b_folder_path}\n"
                f"已尝试的路径: {[str(p) for p in unique_paths[:5]]}\n"
                f"当前工作目录: {current_dir}\n"
                f"请确保文件夹已上传到 uploaded_inputs 目录"
            )
    
    if not m_dir.exists() or not m_dir.is_dir():
        current_dir = Path.cwd()
        possible_paths = []
        
        # 1. 原始路径（已检查，不存在）
        possible_paths.append(m_dir)
        
        # 2. 如果是相对路径，尝试相对于当前工作目录
        if not m_dir.is_absolute():
            possible_paths.append(current_dir / m_dir)
        
        # 3. 如果路径包含 "uploaded_inputs"，尝试在当前目录下查找
        path_str = str(m_dir)
        if "uploaded_inputs" in path_str:
            # 提取 uploaded_inputs 之后的部分
            parts = path_str.split("uploaded_inputs")
            if len(parts) > 1:
                rel_part = parts[1].lstrip("/\\")
                possible_paths.append(current_dir / "uploaded_inputs" / rel_part)
        
        # 4. 尝试从路径中提取目录名
        if m_dir.name:
            possible_paths.append(current_dir / "uploaded_inputs" / m_dir.name)
            possible_paths.append(current_dir / m_dir.name)
        
        # 5. 尝试直接在当前目录的 uploaded_inputs 下查找（递归搜索）
        uploaded_dir = current_dir / "uploaded_inputs"
        if uploaded_dir.exists():
            for subdir in uploaded_dir.iterdir():
                if subdir.is_dir() and subdir.name == m_dir.name:
                    possible_paths.append(subdir)
        
        # 去重并检查
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
                f"M 模式图像文件夹不存在或不是文件夹: {m_folder_path}\n"
                f"已尝试的路径: {[str(p) for p in unique_paths[:5]]}\n"
                f"当前工作目录: {current_dir}\n"
                f"请确保文件夹已上传到 uploaded_inputs 目录"
            )

    pipeline = _get_pipeline()

    output_dir, results_df = pipeline.run(
        b_input=str(b_dir),
        m_input=str(m_dir),
        is_folder=True,
        use_csv=False,
    )

    if results_df is None or len(results_df) == 0:
        raise RuntimeError(
            "批量推理未得到任何结果，请检查 B/M 文件夹下文件名是否匹配且满足命名规则。"
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
# 工具 2：B/M 文件夹批量检测（MCP 对外暴露的入口）
# ============================================================


@mcp.tool(
    name="detect_batch_folders",
    description=(
        "批量检测：输入一个 B 模式图像文件夹和一个 M 模式图像文件夹，"
        "流水线会自动按文件名中的 `YY-MM-DD-<ID>` 进行配对与合并，并返回每个样本的患病概率。"
    ),
)
async def detect_batch_folders(
    b_folder_path: Annotated[
        str,
        Field(
            description=(
                "B 模式超声图像所在文件夹路径。文件夹中可以包含多张图片，"
                "文件名需符合 `YY-MM-DD-<ID>` 规则。"
            )
        ),
    ],
    m_folder_path: Annotated[
        str,
        Field(
            description=(
                "M 模式超声图像所在文件夹路径。文件夹中可以包含多张图片，"
                "文件名需符合 `YY-MM-DD-<ID>` 规则。"
            )
        ),
    ],
) -> BatchDetectionResult:
    """
    多患者 / 多次检查的批量推理接口（MCP 工具包装）。

    内部调用 `detect_batch_folders_impl`，方便本地脚本和 MCP 共用同一套实现。
    """
    return await detect_batch_folders_impl(
        b_folder_path=b_folder_path,
        m_folder_path=m_folder_path,
    )


# ============================================================
# 本地直接运行（非必须，用于调试 MCP 服务器）
# ============================================================


if __name__ == "__main__":
    # 在命令行下直接启动 MCP 服务器，例如：
    #   fastmcp dev agent_et_mcp.py
    # 或直接：
    #   python agent_et_mcp.py
    mcp.run()


