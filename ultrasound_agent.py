"""
使用 Qwen3-8B + FastMCP + 你的 MCP 工具（agent_et_mcp.py）的 Agent 示例。

设计思路：
1. 还是用你现有的 MCP 工具负责“算概率”（detect_single_pair / detect_batch_folders）；
2. Qwen 既可以只负责“看 JSON 结果 + 用中文解释和总结”（解释模式），
   也可以通过 function-calling 的方式主动决定调用哪个 MCP 工具（Agent 模式）。

本文件内提供两部分能力：
- qwen_explain_detection_sync：只做“看 JSON + 说人话”，给 Streamlit 等直接调用；
- run_qwen_agent：让 Qwen 自己根据用户输入选择并调用 MCP 工具，然后再总结结果（真正的 Agent 行为）。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI
import shutil
import time
from typing import Optional


# ============================================================
# 环境变量 & Qwen 客户端初始化
# ============================================================

DEFAULT_QWEN_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"


def get_qwen_client(api_key: str, base_url: str = DEFAULT_QWEN_BASE_URL) -> OpenAI:
    """
    获取一个 OpenAI-兼容客户端，用于调用 Qwen3-8B（SiliconFlow/OpenAI兼容接口）。

    注意：
    - 不要把 API key 写死在代码里；建议用环境变量或 Streamlit secrets/输入框传入。
    """
    return OpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# MCP client reuse & tool discovery
# ============================================================

# 全局复用的 MCP Client（避免每次调用都重启子进程）
_mcp_client: Optional[Client] = None
_mcp_client_entry: Optional[str] = None

# 记录上一次工具产生的 detect_output_dir（用于清理）
_last_detect_output_dirs: List[str] = []


async def get_mcp_client(mcp_entry: str = "agent_et_mcp.py") -> Client:
    """
    获取一个可复用的 FastMCP Client 实例（单例）。

    - 如果已有客户端且 entry 相同则直接返回；
    - 否则关闭旧客户端并创建新客户端。
    """
    global _mcp_client, _mcp_client_entry
    if _mcp_client is not None and _mcp_client_entry == mcp_entry:
        return _mcp_client

    # 关闭旧 client（如果有）
    if _mcp_client is not None:
        try:
            await _mcp_client.__aexit__(None, None, None)
        except Exception:
            pass
        _mcp_client = None
        _mcp_client_entry = None

    # 创建并 enter
    _mcp_client = Client(mcp_entry)
    await _mcp_client.__aenter__()
    _mcp_client_entry = mcp_entry
    return _mcp_client


async def close_mcp_client() -> None:
    """显式关闭全局 MCP client（可选）。"""
    global _mcp_client, _mcp_client_entry
    if _mcp_client is not None:
        try:
            await _mcp_client.__aexit__(None, None, None)
        except Exception:
            pass
    _mcp_client = None
    _mcp_client_entry = None


async def mcp_list_tools(mcp_entry: str = "agent_et_mcp.py") -> List[Dict[str, Any]]:
    """
    尝试通过正在运行的 MCP client 获取工具定义（优先）；如果不可用，则回退到解析 mcp_entry 文件。

    返回：一个适配 Qwen tools 格式的列表（与 QWEN_TOOLS 兼容）。
    """
    # 1) 尝试通过 client.list_tools()（如果 FastMCP 支持）
    try:
        client = await get_mcp_client(mcp_entry)
        if hasattr(client, "list_tools"):
            tools = await client.list_tools()
            # 将 MCP 的 tool 描述映射为 OpenAI/Qwen 的函数工具格式
            qwen_tools: List[Dict[str, Any]] = []
            try:
                for t in tools:
                    # 支持 dict 或对象两种形式
                    if isinstance(t, dict):
                        name = t.get("name") or t.get("tool_name")
                        desc = t.get("description") or t.get("desc") or ""
                        params = t.get("parameters") or t.get("schema") or t.get("args")
                    else:
                        name = getattr(t, "name", None) or getattr(t, "tool_name", None)
                        desc = getattr(t, "description", None) or getattr(t, "desc", None) or ""
                        params = getattr(t, "parameters", None) or getattr(t, "schema", None) or getattr(t, "args", None)

                    if not name:
                        continue

                    func: Dict[str, Any] = {"name": name, "description": desc or ""}

                    # 构建 parameters 的 JSON Schema（最小兼容）
                    parameters_schema: Dict[str, Any]
                    if isinstance(params, dict):
                        # 如果已经是 JSON Schema，直接使用
                        parameters_schema = params
                    elif isinstance(params, list):
                        props: Dict[str, Any] = {}
                        required: List[str] = []
                        for p in params:
                            if isinstance(p, dict):
                                pname = p.get("name")
                                ptype = p.get("type", "string")
                                pdesc = p.get("description", "")
                                preq = bool(p.get("required", False))
                            else:
                                pname = getattr(p, "name", None)
                                ptype = getattr(p, "type", "string")
                                pdesc = getattr(p, "description", "")
                                preq = bool(getattr(p, "required", False))
                            if not pname:
                                continue
                            props[pname] = {"type": ptype, "description": pdesc}
                            if preq:
                                required.append(pname)
                        parameters_schema = {"type": "object", "properties": props}
                        if required:
                            parameters_schema["required"] = required
                    else:
                        # 无法解析参数定义时，保守地提供空 schema
                        parameters_schema = {"type": "object", "properties": {}}

                    func["parameters"] = parameters_schema
                    qwen_tools.append({"type": "function", "function": func})

                if qwen_tools:
                    return qwen_tools
            except Exception:
                # 若映射过程中出现异常，回退到文件解析
                pass
    except Exception:
        # 忽略并回退到文件解析
        pass

    # 2) 回退：解析本地 agent_et_mcp.py 文件，寻找 @mcp.tool 装饰器
    try:
        code = Path(mcp_entry).read_text(encoding="utf-8")
        # 简单正则：抓 name 和 description（如果有）
        pattern = r"@mcp.tool\s*\(\s*name\s*=\s*[\'\"](?P<name>[\w_\-]+)[\'\"](?:,\s*description\s*=\s*[\'\"](?P<desc>.*?)[\'\"])?"
        import re as _re

        qwen_tools = []
        for m in _re.finditer(pattern, code, _re.S):
            name = m.group("name")
            desc = m.group("desc") or ""
            if name == "detect_single_pair":
                qwen_tools.append(QWEN_TOOLS[0])
            elif name == "detect_batch_folders":
                qwen_tools.append(QWEN_TOOLS[1])
        if qwen_tools:
            return qwen_tools
    except Exception:
        pass

    # 3) 最后兜底：返回内置 QWEN_TOOLS
    return QWEN_TOOLS


# ============================================================
# MCP 工具调用封装（沿用你已有的 DetectionPipeline）
# ============================================================


async def mcp_detect_single_pair(
    b_image_path: str,
    m_image_path: str,
    mcp_entry: str = "agent_et_mcp.py",
) -> Dict[str, Any]:
    """
    通过 FastMCP Client 调用 MCP 工具 detect_single_pair，获取单组 B/M 检测结果。
    """
    client = await get_mcp_client(mcp_entry)
    result = await client.call_tool(
        "detect_single_pair",
        {
            "b_image_path": b_image_path,
            "m_image_path": m_image_path,
        },
    )
    return _normalize_mcp_result(result)


async def mcp_detect_batch_folders(
    b_folder_path: str,
    m_folder_path: str,
    mcp_entry: str = "agent_et_mcp.py",
) -> Dict[str, Any]:
    """
    通过 FastMCP Client 调用 MCP 工具 detect_batch_folders，获取批量 B/M 检测结果。
    """
    client = await get_mcp_client(mcp_entry)
    result = await client.call_tool(
        "detect_batch_folders",
        {
            "b_folder_path": b_folder_path,
            "m_folder_path": m_folder_path,
        },
    )
    return _normalize_mcp_result(result)


# ============================================================
# Qwen 负责“看结果 + 说人话”的部分
# ============================================================


QWEN_SYSTEM_PROMPT = """
你是一名资深的重症医学和呼吸康复专家，熟悉基于膈肌 B 模式和 M 模式超声图像的风险评估模型。

你可以使用以下两个工具（已经通过其它模块实现）：
- detect_single_pair：用于“单个患者的一组 B + M 图像”检测，参数：
  - b_image_path：B 模式图片路径（字符串）
  - m_image_path：M 模式图片路径（字符串）
- detect_batch_folders：用于“多患者 B/M 文件夹批量检测”，参数：
  - b_folder_path：B 模式图片文件夹路径
  - m_folder_path：M 模式图片文件夹路径

这两个工具的检测方式都是 B/M 图像组合综合检测，并没有独立检测功能。因此只要有一个缺失就没有检测结果
如果没有完整的 B 和 M 图像组合，工具将无法进行检测。因此对于缺失模态的情况，应在报告中单独说明，在解释时要注意这类病人是没有预测结果的。

当你已经拿到 JSON 结构的“自动检测结果”时，其典型内容包括：
- 对于单个患者：merged_key、risk_probability、prediction、prediction_label 等字段；
- 对于多个患者：上述字段的列表、总样本数、平均风险等信息。

其中 merged_key 的格式为：
- 形如 "YY-MM-DD-C123"、"YY-MM-DD-B123" 或 "YY-MM-DD-P123"；
- 前面的 "YY-MM-DD" 表示检查日期（20YY 年 MM 月 DD 日）；
- 后面的 "C123/B123/P123" 表示患者 ID。

当同一个患者 ID（例如 C113）在不同日期（例如 25-11-08 和 25-12-01）均出现时，表示该患者存在“复检”或“多次随访”。

你的任务：
1. 用清晰、专业但尽量易懂的中文解释这些检测结果的含义；
2. 根据 risk_probability 对风险进行分级（如很低、较低、中等、较高、很高），并给出合理的阈值说明；
3. 如果是批量结果：
   - 指出明显高风险的患者（例如 risk_probability > 0.7），并列出他们的 ID、检查日期和概率；
   - 特别关注同一患者 ID 在不同日期的多次检查（复检情况），对这些患者做“纵向随访式”的综合分析，比较不同日期的风险变化趋势；
4. 如果检测结果中存在“缺失模态”的情况（例如只有 B 没有 M，或只有 M 没有 B），应在报告中单独说明：
   - 说明缺失的是哪一种模态（B 或 M）；
   - 注意，本项目未提供单独的模态检测，因此在解释时要注意这类病人是没有预测结果的。
5. 在给出结论时，至少给出 1~3 条临床或管理上的建议（例如是否需要进一步检查、复查、随访、康复训练或临床评估等）；
6. 明确提醒：这是基于图像的机器学习模型结果，不能替代医生的最终诊断，最终结论需要结合临床情况由医生判断。
"""

QWEN_SYSTEM_PROMPT_EN = """
You are an experienced critical care and respiratory rehabilitation specialist who understands risk assessment
based on diaphragm ultrasound B-mode and M-mode images.

You can use the following tools (already implemented by other modules):
- detect_single_pair: for a single patient with one B-mode and one M-mode image.
- detect_batch_folders: for batch inference with B/M image folders.

These tools only work on paired B/M inputs and do not support single-modality inference.
"""


def qwen_explain_detection_sync(
    detection_json: Dict[str, Any],
    user_intent: str,
    api_key: str,
    base_url: str = DEFAULT_QWEN_BASE_URL,
    model: str = DEFAULT_QWEN_MODEL,
    temperature: float = 0.4,
) -> str:
    """
    同步调用 Qwen3-8B，对检测 JSON 做中文解释。

    参数：
      - detection_json: 从 MCP 工具拿到的检测结果（单个或批量）
      - user_intent: 用户想知道的内容（例如“请解释这名患者的风险并给建议”）
    """
    # 把 JSON 格式化成字符串给模型看
    json_str = json.dumps(detection_json, ensure_ascii=False, indent=2)

    user_prompt = f"""
下面是自动检测模型给出的 JSON 结果：

{json_str}

用户问题/需求如下：
{user_intent}

请按照系统提示中的要求，给出一段完整的中文解读。
"""

    client = get_qwen_client(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    # Qwen-OpenAI 接口风格：choices[0].message.content
    return resp.choices[0].message.content or ""


async def qwen_explain_detection(
    detection_json: Dict[str, Any],
    user_intent: str,
    api_key: str,
    base_url: str = DEFAULT_QWEN_BASE_URL,
    model: str = DEFAULT_QWEN_MODEL,
    temperature: float = 0.4,
) -> str:
    """
    异步包装：在需要 async 的场景下使用（内部仍是同步 HTTP 请求）。
    """
    return qwen_explain_detection_sync(
        detection_json=detection_json,
        user_intent=user_intent,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

# ============================================================
# 示例：让 Qwen 自己决定何时、如何调用 MCP 工具（Agent 行为）
# ============================================================


QWEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_single_pair",
            "description": "对单个患者的一组 B 模式图像和 M 模式图像进行自动特征提取和风险预测。",
            "parameters": {
                "type": "object",
                "properties": {
                    "b_image_path": {
                        "type": "string",
                        "description": "B 模式超声图像的本地路径（绝对或相对）。",
                    },
                    "m_image_path": {
                        "type": "string",
                        "description": "M 模式超声图像的本地路径（绝对或相对）。",
                    },
                },
                "required": ["b_image_path", "m_image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_batch_folders",
            "description": "对多个患者的 B/M 图像文件夹进行批量检测。",
            "parameters": {
                "type": "object",
                "properties": {
                    "b_folder_path": {
                        "type": "string",
                        "description": "B 模式图像文件夹路径。",
                    },
                    "m_folder_path": {
                        "type": "string",
                        "description": "M 模式图像文件夹路径。",
                    },
                },
                "required": ["b_folder_path", "m_folder_path"],
            },
        },
    },
]


def _parse_merged_key(merged_key: str) -> tuple[str | None, str | None]:
    """
    解析 merged_key，例如 "25-11-08-C113" -> ("25-11-08", "C113")
    """
    if not isinstance(merged_key, str):
        return None, None
    m = re.match(r"(\d{2}-\d{2}-\d{2})-(C\d{3}|B\d{3}|P\d{3})", merged_key)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _build_detection_summary_from_tool_result(tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据 MCP 工具返回的结果（单个或批量），构造一个结构化的总结 JSON，
    包含：按患者 ID 的汇总信息、复检情况、以及缺失模态统计（如有）。
    """
    # 1) 统一整理 items 列表
    if "items" in tool_result:
        mode = "batch"
        raw_items: List[Dict[str, Any]] = list(tool_result.get("items") or [])
    else:
        mode = "single"
        raw_items = [tool_result]

    items_enriched: List[Dict[str, Any]] = []
    for r in raw_items:
        merged_key = str(r.get("merged_key", "") or r.get("merged_filename", ""))
        date_str, pid = _parse_merged_key(merged_key)
        items_enriched.append(
            {
                "merged_key": merged_key,
                "date": date_str,
                "patient_id": pid,
                "risk_probability": float(r.get("risk_probability", 0.0)),
                "prediction": int(r.get("prediction", 0)),
                "prediction_label": str(r.get("prediction_label", "")),
                "b_image": str(r.get("b_image", r.get("b_filename", ""))),
                "m_image": str(r.get("m_image", r.get("m_filename", ""))),
            }
        )

    # 2) 识别复检患者（同一 patient_id 多个不同日期）
    visits_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for it in items_enriched:
        pid = it.get("patient_id")
        if not pid:
            continue
        visits_by_pid.setdefault(pid, []).append(it)

    recheck_patients: List[Dict[str, Any]] = []
    for pid, visits in visits_by_pid.items():
        dates = sorted({v.get("date") for v in visits if v.get("date")})
        if len(dates) > 1:
            # 多个日期，视为复检
            recheck_patients.append(
                {
                    "patient_id": pid,
                    "exam_dates": dates,
                    "visits": sorted(
                        visits, key=lambda x: (str(x.get("date") or ""), x.get("merged_key", ""))
                    ),
                }
            )

    # 3) 缺失模态统计（依赖 missing_modality_samples.csv，如存在）
    missing_summary: Dict[str, Any] | None = None
    detect_output_dir = tool_result.get("detect_output_dir")
    if isinstance(detect_output_dir, str) and detect_output_dir:
        missing_csv_path = os.path.join(detect_output_dir, "missing_modality_samples.csv")
        if os.path.exists(missing_csv_path):
            try:
                df_missing = pd.read_csv(missing_csv_path)
                total_missing = int(len(df_missing))
                by_type = (
                    df_missing["missing_modality"].value_counts().to_dict()
                    if "missing_modality" in df_missing.columns
                    else {}
                )
                by_patient: Dict[str, int] = {}
                if "pid" in df_missing.columns:
                    by_patient = df_missing["pid"].value_counts().to_dict()
                missing_summary = {
                    "total_missing_samples": total_missing,
                    "missing_by_type": by_type,
                    "missing_by_patient": by_patient,
                    "csv_path": missing_csv_path,
                }
            except Exception:
                # 读取失败则忽略缺失统计，避免中断主流程
                missing_summary = None

    # 4) 汇总整体信息
    all_probs = [it["risk_probability"] for it in items_enriched]
    avg_prob = float(sum(all_probs) / len(all_probs)) if all_probs else 0.0

    summary: Dict[str, Any] = {
        "mode": mode,
        "total_samples": len(items_enriched),
        "average_probability": avg_prob,
        "items": items_enriched,
        "recheck_patients": recheck_patients,
    }
    if missing_summary is not None:
        summary["missing_modality_summary"] = missing_summary

    return summary


async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    由 Agent 示例使用：根据工具名和参数真正调用 MCP 工具。
    这里直接使用 FastMCP Client 的通用调用方式。
    """
    # 首先尝试使用复用的全局 client（提高性能，避免频繁重启子进程）
    try:
        client = await get_mcp_client("agent_et_mcp.py")
        try:
            result = await client.call_tool(tool_name, arguments)
        except RuntimeError:
            # 如果 client 尚未连接或 session 无效，回退到短生命周期的 context 调用
            client = None
        else:
            normalized = _normalize_mcp_result(result)
            # 记录 detect_output_dir，便于后续清理历史检测输出
            try:
                d = normalized.get("detect_output_dir")
                if isinstance(d, str) and d:
                    if d not in _last_detect_output_dirs:
                        _last_detect_output_dirs.append(d)
            except Exception:
                pass
            return normalized
    except Exception:
        # 任何异常都回退到短生命周期 context 调用
        client = None

    # 如果复用 client 不可用，则使用短生命周期的 async with Client(...) 调用
    async with Client("agent_et_mcp.py") as transient_client:
        result = await transient_client.call_tool(tool_name, arguments)
        normalized = _normalize_mcp_result(result)
    # 记录 detect_output_dir，便于后续清理历史检测输出
        try:
            d = normalized.get("detect_output_dir")
            if isinstance(d, str) and d:
                if d not in _last_detect_output_dirs:
                    _last_detect_output_dirs.append(d)
        except Exception:
            pass
        return normalized


async def _cleanup_previous_detect_outputs() -> None:
    """删除上一次 run 中记录的 detect_output_dir（如果存在），并清空记录。"""
    global _last_detect_output_dirs
    if not _last_detect_output_dirs:
        return
    for d in list(_last_detect_output_dirs):
        try:
            if d and os.path.exists(d) and os.path.isdir(d):
                shutil.rmtree(d)
        except Exception:
            # 忽略删除失败，继续尝试其它目录
            pass
    _last_detect_output_dirs = []


def export_agent_conversation(messages: List[Dict[str, Any]],
                              tool_calls: List[Dict[str, Any]],
                              tool_results: Dict[str, Any],
                              final_response: str,
                              out_dir: str = "detect") -> str:
    """
    导出 agent 的完整对话记录（messages + tool_calls + tool_results + final_response）到 JSON 文件，返回文件路径。
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        path = os.path.join(out_dir, f"agent_conversation_{ts}.json")
        payload = {
            "messages": messages,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "final_response": final_response,
            "exported_at": ts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return ""


def _normalize_mcp_result(result: Any) -> Dict[str, Any]:
    """
    将 FastMCP / MCP 工具的返回值标准化为 Python dict。

    支持的输入示例：
      - list[...]（包含 TextContent 对象，其 .text 字段是 JSON 字符串）
      - dict（已经是标准 dict）
      - CallToolResult（具有 structured_content / content / data 属性）
      - 其它可字符串化的对象

    返回：若能解析为 JSON 或 structured_content，则返回对应 dict；否则返回 {"raw": str(result)}。
    """
    # 1) 直接的 list（常见于 fastmcp 返回）
    try:
        if isinstance(result, list) and result:
            first = result[0]
            # 某些实现中 element 有 .text
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
            # 或者 element 本身就是 dict-like
            if isinstance(first, dict):
                return first

        # 2) 直接是 dict
        if isinstance(result, dict):
            return result

        # 3) FastMCP 的 CallToolResult 可能带有 structured_content 属性
        structured = getattr(result, "structured_content", None)
        if structured:
            try:
                # structured_content 通常已经是 dict
                return dict(structured)
            except Exception:
                return {"raw_structured": str(structured)}

        # 4) content 列表也常见：取第一个元素的 .text
        content = getattr(result, "content", None)
        if content and len(content) > 0:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}

        # 5) data 字段（有时为 pydantic 模型或类似对象）
        data = getattr(result, "data", None)
        if data is not None:
            # 如果是 pydantic BaseModel
            try:
                if hasattr(data, "dict"):
                    return data.dict()
            except Exception:
                pass
            try:
                return dict(data)
            except Exception:
                return {"raw_data": str(data)}

    except Exception:
        # 避免在解析器内抛出异常，统一回退为 raw 字符串
        return {"raw": str(result)}

    # 最后兜底：返回原始字符串
    return {"raw": str(result)}


async def run_qwen_agent(
    b_image_path: str = None,
    m_image_path: str = None,
    b_folder_path: str = None,
    m_folder_path: str = None,
    api_key: str = None,
    base_url: str = DEFAULT_QWEN_BASE_URL,
    model: str = DEFAULT_QWEN_MODEL,
    user_query: str = None,
    language: str = "中文",
) -> Dict[str, Any]:
    """
    真正的 Agent 行为——让 Qwen 自己“决定调用哪个检测工具”。

    参数：
      - b_image_path, m_image_path: 单组检测的图片路径
      - b_folder_path, m_folder_path: 批量检测的文件夹路径
      - api_key: Qwen API Key
      - base_url: API base URL（默认 SiliconFlow）
      - model: 模型名称（默认 Qwen3-8B）
      - user_query: 用户自然语言需求（可选，会自动生成）
    - language: 输出语言（"中文" 或 "English"）

    返回：
      {
        "tool_calls": [...],  # 工具调用记录
        "tool_results": {...},  # 工具返回结果
        "final_response": "...",  # Qwen 最终回答
      }
    """
    if not api_key:
        raise ValueError("必须提供 api_key")

    # 语言控制（默认为中文）
    lang = (language or "中文").strip().lower()
    is_english = lang in ["en", "english", "英文"]
    lang_instruction = "Please respond in English." if is_english else "请使用中文回答。"

    # 自动生成 user_query（如果未提供）
    if not user_query:
        if b_image_path and m_image_path:
            if is_english:
                user_query = f"""
I have one patient's diaphragm ultrasound images. Please call the appropriate tool:
- B-mode image path: {b_image_path}
- M-mode image path: {m_image_path}

Please:
1. Do NOT provide conclusions before calling the tool; you must call the MCP tool first.
2. Choose and call the correct detection tool based on the paths above.
3. After receiving JSON results, explain the patient's risk probability and risk level.
4. Provide 1-3 clinical recommendations in English.
5. Remind that this is model output and not a medical diagnosis.
6. If B/M paths are provided, you must call the tool.
7. Do not guess risk before tool call.
8. Tool call is mandatory, not optional.
"""
            else:
                user_query = f"""
我这边有一名患者的一组膈肌超声图像，请你根据需要调用合适的检测工具：
- B 模式图片路径：{b_image_path}
- M 模式图片路径：{m_image_path}

请你：
1. 没有调用工具之前不要给出任何结论，你必须要真正的调用MCP工具才能继续回答；
2. 先根据这些路径，选择并调用正确的自动检测工具；
3. 拿到检测 JSON 结果后，说明该患者的患病风险概率和风险级别；
4. 给出 1~3 条中文的临床建议；
5. 提醒这是模型预测结果，不能替代医生诊断。
6. 只要用户提供了 B/M 图像路径，就必须优先调用检测工具；
7. 在未调用工具前，不得基于假设给出风险判断；
8. 工具调用是强制步骤，而不是可选步骤；
9. 可选步骤是根据用户输入的路径选择调用哪个工具。
10.不能虚假的调用工具，用户能够看到你是否真正调用了工具。
"""
        elif b_folder_path and m_folder_path:
            if is_english:
                user_query = f"""
I have folders with multiple patients' diaphragm ultrasound images. Please call the batch tool:
- B-mode folder path: {b_folder_path}
- M-mode folder path: {m_folder_path}

Please:
1. Do NOT provide conclusions before calling the tool; you must call the MCP tool first.
2. Call the batch tool (detect_batch_folders).
3. After receiving JSON results, summarize the risk distribution across patients.
4. Highlight high-risk patients (e.g., risk_probability > 0.7) with their IDs and probabilities.
5. Provide 1-3 clinical or management recommendations in English.
6. Remind that this is model output and not a medical diagnosis.
"""
            else:
                user_query = f"""
我这边有多个患者的膈肌超声图像文件夹，请你根据需要调用合适的批量检测工具：
- B 模式图片文件夹路径：{b_folder_path}
- M 模式图片文件夹路径：{m_folder_path}

请你：
1. 没有调用工具之前不要给出任何结论，你必须要真正的调用MCP工具才能继续回答；
2. 先根据这些路径，选择并调用正确的批量检测工具（detect_batch_folders）；
3. 拿到检测 JSON 结果后，统计并说明所有患者的患病风险分布；
4. 指出高风险患者（例如 risk_probability > 0.7）并列出他们的 ID 和概率；
5. 给出 1~3 条临床或管理建议；
6. 提醒这是模型预测结果，不能替代医生诊断。
"""
        else:
            raise ValueError("必须提供 (b_image_path, m_image_path) 或 (b_folder_path, m_folder_path)")

    if user_query:
        user_query = user_query.strip() + "\n" + lang_instruction

    # 先清理上一次检测遗留的输出（如果有）
    await _cleanup_previous_detect_outputs()

    # 获取 Qwen client
    client = get_qwen_client(api_key=api_key, base_url=base_url)

    system_prompt = QWEN_SYSTEM_PROMPT_EN if is_english else QWEN_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # 尝试动态从 MCP 获取工具定义，优先使用运行中的 MCP
    try:
        tools = await mcp_list_tools("agent_et_mcp.py")
    except Exception:
        tools = QWEN_TOOLS

    # 步骤1：让 Qwen 决定调用哪个工具（将 tools 动态传入）
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
    )

    choice = first.choices[0]
    assistant_msg = choice.message
    messages.append(
        {
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": assistant_msg.tool_calls,
        }
    )

    tool_calls_info = []
    tool_results_dict = {}
    detection_summary: Dict[str, Any] | None = None

    # 如果没有 tool_calls，直接返回模型回答
    if not assistant_msg.tool_calls:
        return {
            "tool_calls": [],
            "tool_results": {},
            "final_response": assistant_msg.content or "",
        }

    # 步骤2：执行每个工具调用
    for tool_call in assistant_msg.tool_calls:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}

        tool_calls_info.append({"name": tool_name, "arguments": args})

        tool_result = await _call_mcp_tool(tool_name, args)
        tool_results_dict[tool_name] = tool_result

        # 把工具结果作为 tool 消息加入对话历史
        messages.append(
            {
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", None),
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

        # 顺便尝试构建一个结构化的汇总（只需构建一次即可）
        if detection_summary is None and isinstance(tool_result, dict):
            try:
                detection_summary = _build_detection_summary_from_tool_result(tool_result)
            except Exception:
                detection_summary = None

    # 如果成功构建了结构化汇总，把它作为额外提示信息给到 Qwen，
    # 明确请求在最终回答中参考其中的复检信息和缺失模态信息。
    if detection_summary is not None:
        summary_str = json.dumps(detection_summary, ensure_ascii=False, indent=2)
        messages.append(
            {
                "role": "user",
                "content": (
                    "下面是系统根据检测工具返回结果整理出的结构化总结 JSON，"
                    "其中已经提取了每个样本的 merged_key（检查日期 + 患者 ID）、"
                    "复检患者列表（同一 ID 在不同日期的多次检查）、以及缺失模态统计信息：\n\n"
                    f"{summary_str}\n\n"
                    "在给出最终中文报告时，请务必：\n"
                    "1）正确理解 merged_key 中的检查日期和患者 ID；\n"
                    "2）特别指出有哪些患者存在复检，并比较不同日期之间的风险变化；\n"
                    "3）如果存在缺失模态（missing modality），在报告中单独说明这一点及其对结论的不确定性影响；\n"
                    "4）其余要求仍然按系统提示中的说明执行。"
                ),
            }
        )

    # 步骤3：让 Qwen 基于工具结果和结构化总结给出最终回答
    second = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    final_msg = second.choices[0].message
    # 如果模型没有生成文本回答，收集调试信息以便排查
    def _sanitize_for_json(x):
        # 递归将复杂对象转换为 JSON-可序列化的形式（不可识别的对象将被 str()）
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, dict):
            return {str(k): _sanitize_for_json(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize_for_json(v) for v in x]
        try:
            # 尝试直接序列化常见类型
            return json.loads(json.dumps(x))
        except Exception:
            try:
                return str(x)
            except Exception:
                return "<unserializable>"

    final_text = final_msg.content or ""

    # 导出对话记录（可供下载）
    convo_path = export_agent_conversation(messages, tool_calls_info, tool_results_dict, final_text)

    result_obj = {
        "tool_calls": tool_calls_info,
        "tool_results": tool_results_dict,
        "final_response": final_text,
        "conversation_export_path": convo_path,
    }

    if not final_text.strip():
        # 补充可读的 debug 信息
        debug = {
            "messages": _sanitize_for_json(messages),
            "assistant_msg_tool_calls": _sanitize_for_json(getattr(choice, 'message', None) and getattr(choice.message, 'tool_calls', None)),
            "first_choice": _sanitize_for_json(choice),
            "second_choices": _sanitize_for_json(getattr(second, 'choices', None)),
            "tool_results": _sanitize_for_json(tool_results_dict),
        }
        result_obj["debug"] = debug

    return result_obj


async def run_agent_example() -> None:
    """
    命令行示例：运行 Qwen Agent。
    """
    b_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\B_model_one\25-08-11-B013_25-08-11-B013-L-Tdi-exp1.jpg"
    m_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\M_model_one\25-08-11-B013_25-08-11-B013-R-DE-DB1.jpg"

    if not Path(b_image_path).exists() or not Path(m_image_path).exists():
        print("❌ 示例 B/M 图片路径不存在，请先修改 deepseek_ultrasound_agent.py 中的路径。")
        return

    api_key = "sk-jxf"

    print("==============================================")
    print("  运行 Qwen Agent（让 Qwen 自己调用工具）")
    print("==============================================")

    result = await run_qwen_agent(
        b_image_path=b_image_path,
        m_image_path=m_image_path,
        api_key=api_key,
    )

    print("\n工具调用记录：")
    for tc in result["tool_calls"]:
        print(f"  - {tc['name']} with args={tc['arguments']}")

    print("\n工具返回结果（截断）：")
    for name, res in result["tool_results"].items():
        print(f"  {name}: {json.dumps(res, ensure_ascii=False, indent=2)[:300]}...")

    print("\n******** Qwen Agent 最终回答 ********\n")
    print(result["final_response"])
    print("\n****************************************\n")


async def main() -> None:
    """
    命令行入口：
    - 你可以在这里选择跑“纯解释模式”还是“Agent 模式”。
    """
    # 让 Qwen 亲自调用工具（真正 Agent 行为）
    await run_agent_example()


if __name__ == "__main__":
    asyncio.run(main())


