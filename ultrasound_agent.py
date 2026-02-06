"""
Agent construction using Qwen3-8B + FastMCP +  MCP tools (agent_et_mcp.py).

- run_qwen_agent: let Qwen decide which MCP tool to call and then summarize the result (true agent behavior).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List
import importlib.util
import inspect
import sys

import pandas as pd
from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI
import shutil
import time
from typing import Optional


# ============================================================
# Environment variables & Qwen client initialization
# ============================================================

DEFAULT_QWEN_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"


def get_qwen_client(api_key: str, base_url: str = DEFAULT_QWEN_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# MCP client reuse & tool discovery
# ============================================================

# Reuse a global MCP client (avoid restarting subprocesses per call)
_mcp_client: Optional[Client] = None
_mcp_client_entry: Optional[str] = None


def _resolve_mcp_entry(mcp_entry: str) -> str:
    """Resolve MCP entry path to an absolute path based on this file's directory."""
    entry_path = Path(mcp_entry)
    if entry_path.is_absolute():
        return str(entry_path)
    base_dir = Path(__file__).resolve().parent
    return str((base_dir / entry_path).resolve())

# Track detect_output_dir from previous tool calls (for cleanup)
_last_detect_output_dirs: List[str] = []


async def get_mcp_client(mcp_entry: str = "agent_et_mcp.py") -> Client:
    """
    Get a reusable FastMCP client instance.

    - If an existing client with the same entry exists, return it.
    - Otherwise close the old client and create a new one.
    """
    global _mcp_client, _mcp_client_entry
    resolved_entry = _resolve_mcp_entry(mcp_entry)
    if _mcp_client is not None and _mcp_client_entry == resolved_entry:
        # Ensure the cached client is still connected
        try:
            _ = _mcp_client.session
            return _mcp_client
        except RuntimeError:
            # Session is invalid; drop the client without awaiting a close
            _mcp_client = None
            _mcp_client_entry = None

    # Close the old client (if any)
    if _mcp_client is not None:
        try:
            await _mcp_client.__aexit__(None, None, None)
        except (Exception, asyncio.CancelledError):
            pass
        _mcp_client = None
        _mcp_client_entry = None

    # Create and enter
    _mcp_client = Client(resolved_entry)
    await _mcp_client.__aenter__()
    _mcp_client_entry = resolved_entry
    return _mcp_client


async def close_mcp_client() -> None:
    """Explicitly close the global MCP client."""
    global _mcp_client, _mcp_client_entry
    if _mcp_client is not None:
        try:
            await _mcp_client.__aexit__(None, None, None)
        except (Exception, asyncio.CancelledError):
            pass
    _mcp_client = None
    _mcp_client_entry = None


async def mcp_list_tools(mcp_entry: str = "agent_et_mcp.py") -> List[Dict[str, Any]]:
    """
    Obtain tool definitions from the MCP server and reconnect if the client is disconnected.

    Returns: a list compatible with Qwen tool format.
    """
    def _to_qwen_tools(tools: List[Any]) -> List[Dict[str, Any]]:
        qwen_tools: List[Dict[str, Any]] = []
        for t in tools:
            # Support dict or object form
            if isinstance(t, dict):
                name = t.get("name") or t.get("tool_name")
                desc = t.get("description") or t.get("desc") or ""
                params = (
                    t.get("parameters")
                    or t.get("schema")
                    or t.get("input_schema")
                    or t.get("inputSchema")
                    or t.get("args")
                )
            else:
                name = getattr(t, "name", None) or getattr(t, "tool_name", None)
                desc = getattr(t, "description", None) or getattr(t, "desc", None) or ""
                params = (
                    getattr(t, "parameters", None)
                    or getattr(t, "schema", None)
                    or getattr(t, "input_schema", None)
                    or getattr(t, "inputSchema", None)
                    or getattr(t, "args", None)
                )

            if not name:
                continue

            func: Dict[str, Any] = {"name": name, "description": desc or ""}

            # Build parameters JSON schema (minimal compatibility)
            parameters_schema: Dict[str, Any]
            if isinstance(params, dict):
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
                parameters_schema = {"type": "object", "properties": {}}

            func["parameters"] = parameters_schema
            qwen_tools.append({"type": "function", "function": func})
        return qwen_tools

    # Attempt using cached client first, reconnect on failure
    for _ in range(2):
        try:
            client = await get_mcp_client(mcp_entry)
            if hasattr(client, "list_tools"):
                tools = await client.list_tools()
                qwen_tools = _to_qwen_tools(tools)
                if qwen_tools:
                    return qwen_tools
        except RuntimeError:
            await close_mcp_client()
        except Exception:
            await close_mcp_client()

    # Last resort: short-lived MCP client context (still uses MCP server)
    async with Client(_resolve_mcp_entry(mcp_entry)) as transient_client:
        tools = await transient_client.list_tools()
        qwen_tools = _to_qwen_tools(tools)
        if qwen_tools:
            return qwen_tools

    raise RuntimeError("No tools returned from MCP list_tools()")


def _load_local_tools_from_entry(mcp_entry: str) -> List[Dict[str, Any]]:
    """
    Load tool definitions directly from the MCP entry module without starting a client.
    This is a safe fallback to avoid hard-coded QWEN_TOOLS.
    """
    entry_path = _resolve_mcp_entry(mcp_entry)
    module_name = f"_mcp_tools_{Path(entry_path).stem}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, entry_path)
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        return []

    mcp_obj = getattr(module, "mcp", None)
    candidates: List[Any] = []

    if mcp_obj is not None:
        for attr_name in ("tools", "_tools", "tool_registry", "tool_specs"):
            attr = getattr(mcp_obj, attr_name, None)
            if isinstance(attr, dict):
                candidates.extend(list(attr.values()))
            elif isinstance(attr, list):
                candidates.extend(attr)

    # If we still have nothing, try to infer from known functions
    if not candidates:
        for fn_name in ("detect_single_pair", "detect_batch_folders"):
            fn = getattr(module, fn_name, None)
            if callable(fn):
                candidates.append(fn)

    return _build_qwen_tools_from_candidates(candidates)


def _build_qwen_tools_from_candidates(candidates: List[Any]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, dict):
            name = item.get("name") or item.get("tool_name")
            desc = item.get("description") or item.get("desc") or ""
            params = (
                item.get("parameters")
                or item.get("schema")
                or item.get("input_schema")
                or item.get("inputSchema")
                or item.get("args")
            )
        else:
            name = getattr(item, "name", None) or getattr(item, "tool_name", None) or getattr(item, "__name__", None)
            desc = (
                getattr(item, "description", None)
                or getattr(item, "desc", None)
                or getattr(item, "__doc__", None)
                or ""
            )
            params = (
                getattr(item, "parameters", None)
                or getattr(item, "schema", None)
                or getattr(item, "input_schema", None)
                or getattr(item, "inputSchema", None)
                or getattr(item, "args", None)
            )

        if not name:
            continue

        # If params are missing but we have a callable, use annotations to build a minimal schema
        if params is None and callable(item):
            try:
                sig = inspect.signature(item)
                props: Dict[str, Any] = {}
                required: List[str] = []
                for param in sig.parameters.values():
                    if param.name in ("self", "cls"):
                        continue
                    props[param.name] = {"type": "string", "description": ""}
                    if param.default is inspect.Parameter.empty:
                        required.append(param.name)
                params = {"type": "object", "properties": props}
                if required:
                    params["required"] = required
            except Exception:
                params = {"type": "object", "properties": {}}

        # Normalize into Qwen/OpenAI tool format
        func: Dict[str, Any] = {"name": name, "description": desc or ""}
        if isinstance(params, dict):
            func["parameters"] = params
        else:
            func["parameters"] = {"type": "object", "properties": {}}

        tools.append({"type": "function", "function": func})

    # De-duplicate by name, keeping the first occurrence
    seen = set()
    unique_tools: List[Dict[str, Any]] = []
    for t in tools:
        t_name = t.get("function", {}).get("name")
        if not t_name or t_name in seen:
            continue
        seen.add(t_name)
        unique_tools.append(t)

    return unique_tools


# ============================================================
# MCP tool wrappers
# ============================================================


async def mcp_detect_single_pair(
    b_image_path: str,
    m_image_path: str,
    mcp_entry: str = "agent_et_mcp.py",
) -> Dict[str, Any]:
    """
    Call MCP tool detect_single_pair via FastMCP client to get a single B/M detection result.
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
    Call MCP tool detect_batch_folders via FastMCP client to get batch B/M detection results.
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
# Qwen explanation section
# ============================================================


QWEN_SYSTEM_PROMPT = """
You are a senior critical care and respiratory rehabilitation specialist familiar with risk assessment models based on diaphragm B-mode and M-mode ultrasound images.

You can use the following two tools (implemented in other modules):
- detect_single_pair: for detecting a single patient's B + M image pair, parameters:
    - b_image_path: B-mode image path (string)
    - m_image_path: M-mode image path (string)
- detect_batch_folders: for batch detection across multiple patients' B/M folders, parameters:
    - b_folder_path: B-mode image folder path
    - m_folder_path: M-mode image folder path

Both tools only work on combined B/M image pairs and do not support standalone modality detection. If either modality is missing, there will be no prediction.
When there is no complete B and M pair, the tools cannot run. For missing modality cases, mention them separately and note that no prediction is available.

When you receive the JSON detection result, typical contents include:
- For a single patient: merged_key, risk_probability, prediction, prediction_label, etc.
- For multiple patients: a list of those fields and total samples.

The merged_key format:
- Looks like "YY-MM-DD-C123", "YY-MM-DD-B123", or "YY-MM-DD-P123".
- "YY-MM-DD" is the exam date (20YY-MM-DD).
- "C123/B123/P123" is the patient ID.

If the same patient ID appears on different dates (e.g., 25-11-08 and 25-12-01), it indicates repeat exams or follow-ups.

Your tasks:
1. Explain the results clearly and professionally in English, while keeping them easy to understand.
2. Categorize risk by risk_probability (e.g., very low, low, medium, high, very high) and provide reasonable threshold descriptions.
3. For batch results:
    - Highlight high-risk patients (risk_probability > 0.6) with ID, date, and probability.
    - Pay special attention to repeat exams for the same patient ID and analyze risk trends across dates.
4. If missing modality cases exist (only B or only M), note them explicitly:
    - Specify which modality is missing (B or M).
    - This project does not provide single-modality prediction, so no prediction is available for those cases.
5. Provide at least 1–3 clinical or management suggestions (e.g., further exams, rechecks, follow-ups, rehab training, or clinical evaluation).
6. Clearly remind that this is a machine learning result based on images and cannot replace a doctor's final diagnosis.
7. Structure the output in this order:
    - Sample details summary (brief)
    - High-risk patient analysis
    - High-risk patient suggestions
    - Recheck patient analysis (if any)
    - Recheck patient suggestions (if any)
    - Missing modality analysis (if any)
    - Missing modality suggestions (if any)
    - Risk definition thresholds
    - Missing modality samples summary
    - Disclaimer (model output cannot replace doctor's diagnosis)
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
    Call Qwen3-8B synchronously to explain the detection JSON in English.

    Args:
        - detection_json: detection result from MCP tools (single or batch)
        - user_intent: user intent (e.g., "Please explain the patient's risk and give suggestions")
    """
    # Format JSON as a string for the model
    json_str = json.dumps(detection_json, ensure_ascii=False, indent=2)

    user_prompt = f"""
Below is the JSON result from the automated detection model:

{json_str}

User question/need:
{user_intent}

Please follow the system instructions and provide a complete explanation in English.
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

    # Qwen-OpenAI API style: choices[0].message.content
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
    Async wrapper for use in async contexts (internally still synchronous HTTP calls).
    """
    return qwen_explain_detection_sync(
        detection_json=detection_json,
        user_intent=user_intent,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def _parse_merged_key(merged_key: str) -> tuple[str | None, str | None]:
    if not isinstance(merged_key, str):
        return None, None
    m = re.match(r"(\d{2}-\d{2}-\d{2})-(C\d{3}|B\d{3}|P\d{3})", merged_key)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _build_detection_summary_from_tool_result(tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a structured summary JSON from MCP tool results (single or batch),
    including per-patient summaries, recheck info, and missing modality stats (if any).
    """
    # 1) Normalize items list
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

    # 2) Identify recheck patients (same patient_id across different dates)
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
            # Multiple dates -> recheck
            recheck_patients.append(
                {
                    "patient_id": pid,
                    "exam_dates": dates,
                    "visits": sorted(
                        visits, key=lambda x: (str(x.get("date") or ""), x.get("merged_key", ""))
                    ),
                }
            )

    # 3) Missing modality summary (from missing_modality_samples.csv if present)
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
                # Ignore missing stats if reading fails
                missing_summary = None

    # 4) Overall summary
    summary: Dict[str, Any] = {
        "mode": mode,
        "total_samples": len(items_enriched),
        "items": items_enriched,
        "recheck_patients": recheck_patients,
    }
    if missing_summary is not None:
        summary["missing_modality_summary"] = missing_summary

    return summary


async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    # First try the reused global client (performance; avoid frequent restarts)
    try:
        client = await get_mcp_client("agent_et_mcp.py")
        try:
            result = await client.call_tool(tool_name, arguments)
        except RuntimeError:
            # If the client is not connected or session is invalid, fall back to a short-lived context
            client = None
        else:
            normalized = _normalize_mcp_result(result)
            # Track detect_output_dir for cleanup
            try:
                d = normalized.get("detect_output_dir")
                if isinstance(d, str) and d:
                    if d not in _last_detect_output_dirs:
                        _last_detect_output_dirs.append(d)
            except Exception:
                pass
            return normalized
    except Exception:
        # On any error, fall back to short-lived context
        client = None

    # If the reused client is unavailable, use a short-lived async context client
    async with Client(_resolve_mcp_entry("agent_et_mcp.py")) as transient_client:
        result = await transient_client.call_tool(tool_name, arguments)
        normalized = _normalize_mcp_result(result)
    # Track detect_output_dir for cleanup
        try:
            d = normalized.get("detect_output_dir")
            if isinstance(d, str) and d:
                if d not in _last_detect_output_dirs:
                    _last_detect_output_dirs.append(d)
        except Exception:
            pass
        return normalized


async def _cleanup_previous_detect_outputs() -> None:
    """Delete detect_output_dir recorded from the previous run (if any) and clear the list."""
    global _last_detect_output_dirs
    if not _last_detect_output_dirs:
        return
    for d in list(_last_detect_output_dirs):
        try:
            if d and os.path.exists(d) and os.path.isdir(d):
                shutil.rmtree(d)
        except Exception:
            # Ignore delete failures and continue
            pass
    _last_detect_output_dirs = []


def export_agent_conversation(messages: List[Dict[str, Any]],
                              tool_calls: List[Dict[str, Any]],
                              tool_results: Dict[str, Any],
                              final_response: str,
                              out_dir: str = "detect") -> str:
    """
    Export the agent conversation (messages + tool_calls + tool_results + final_response) to JSON and return the file path.
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
    Normalize FastMCP/MCP tool results into a Python dict.

    Supported inputs:
        - list[...] (contains TextContent objects whose .text is JSON)
        - dict (already a dict)
        - CallToolResult (has structured_content / content / data)
        - other objects convertible to string

    Returns: parsed dict when possible; otherwise {"raw": str(result)}.
    """
    # 1) Direct list (common in fastmcp returns)
    try:
        if isinstance(result, list) and result:
            first = result[0]
            # Some implementations expose .text on the element
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
            # Or the element itself is dict-like
            if isinstance(first, dict):
                return first

        # 2) Direct dict
        if isinstance(result, dict):
            return result

        # 3) CallToolResult may include structured_content
        structured = getattr(result, "structured_content", None)
        if structured:
            try:
                # structured_content is usually already a dict
                return dict(structured)
            except Exception:
                return {"raw_structured": str(structured)}

        # 4) content list is common: use first element's .text
        content = getattr(result, "content", None)
        if content and len(content) > 0:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}

        # 5) data field (sometimes a pydantic model or similar)
        data = getattr(result, "data", None)
        if data is not None:
            # If it's a pydantic BaseModel
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
        # Avoid raising inside the parser; fall back to raw string
        return {"raw": str(result)}

    # Final fallback: return raw string
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
) -> Dict[str, Any]:
    """
    True agent behavior—let Qwen decide which detection tool to call.

    Args:
        - b_image_path, m_image_path: image paths for a single exam
        - b_folder_path, m_folder_path: folder paths for batch exams
        - api_key: Qwen API key
        - base_url: API base URL (default SiliconFlow)
        - model: model name (default Qwen3-8B)
        - user_query: user natural language request (optional; auto-generated)

    Returns:
        {
            "tool_calls": [...],  # tool call records
            "tool_results": {...},  # tool results
            "final_response": "...",  # Qwen final response
        }
    """
    if not api_key:
        raise ValueError("api_key is required")

    # Auto-generate user_query
    if not user_query:
        if b_image_path and m_image_path:
            user_query = f"""
I have a patient's diaphragm ultrasound image pair. Please call the appropriate detection tool as needed:
- B-mode image path: {b_image_path}
- M-mode image path: {m_image_path}

Please:
1. Do not provide any conclusions before calling a tool; you must actually call the MCP tool before responding.
2. Choose and call the correct detection tool based on these paths.
3. After getting the JSON result, explain the patient's risk probability and risk level.
4. Provide 1–3 clinical suggestions in English.
5. Remind that this is a model prediction and cannot replace a doctor's diagnosis.
6. As long as the user provides B/M image paths, you must call the detection tool first.
7. Before calling a tool, do not infer risk based on assumptions.
8. Tool calling is mandatory, not optional.
9. The optional step is choosing which tool to call based on the input paths.
10. Do not fake tool calls; the user can see whether you actually called the tool.
"""
        elif b_folder_path and m_folder_path:
            user_query = f"""
I have folders of diaphragm ultrasound images for multiple patients. Please call the appropriate batch detection tool:
- B-mode image folder path: {b_folder_path}
- M-mode image folder path: {m_folder_path}

Please:
1. Do not provide any conclusions before calling a tool; you must actually call the MCP tool before responding.
2. Choose and call the correct batch detection tool (detect_batch_folders).
3. After getting the JSON result, summarize the risk distribution across patients.
4. Identify high-risk patients (risk_probability > 0.6) and list their IDs and probabilities.
5. Provide 1–3 clinical or management suggestions.
6. Remind that this is a model prediction and cannot replace a doctor's diagnosis.
"""
        else:
            raise ValueError("Provide (b_image_path, m_image_path) or (b_folder_path, m_folder_path)")

    # Clean up outputs from the previous run (if any)
    await _cleanup_previous_detect_outputs()

    # Get Qwen client
    client = get_qwen_client(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    # Try to fetch tools dynamically from MCP, with local fallback inside mcp_list_tools
    tools = await mcp_list_tools("agent_et_mcp.py")

    # Step 1: let Qwen decide which tool to call (dynamic tools list)
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

    # If there are no tool calls, return the model response directly
    if not assistant_msg.tool_calls:
        return {
            "tool_calls": [],
            "tool_results": {},
            "final_response": assistant_msg.content or "",
        }

    # Step 2: execute each tool call
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

        # Add tool result as a tool message in the conversation history
        messages.append(
            {
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", None),
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

        # Try to build a structured summary (only once)
        if detection_summary is None and isinstance(tool_result, dict):
            try:
                detection_summary = _build_detection_summary_from_tool_result(tool_result)
            except Exception:
                detection_summary = None

    # If a structured summary is available, provide it to Qwen as extra context,
    # asking it to reference recheck info and missing modality info in the final response.
    if detection_summary is not None:
        summary_str = json.dumps(detection_summary, ensure_ascii=False, indent=2)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Below is a structured summary JSON generated from the tool results,"
                    " including merged_key (exam date + patient ID), recheck patient list,"
                    " and missing modality statistics:\n\n"
                    f"{summary_str}\n\n"
                    "When producing the final English report, please:\n"
                    "1) Correctly interpret the exam date and patient ID in merged_key;\n"
                    "2) Highlight patients with rechecks and compare risk trends across dates;\n"
                    "3) If missing modality cases exist, mention them explicitly;\n"
                    "4) Follow the remaining system instructions as stated."
                ),
            }
        )

    # Step 3: let Qwen produce the final response based on tool results and summary
    second = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    final_msg = second.choices[0].message
    # If the model returns empty text, collect debug info for troubleshooting
    def _sanitize_for_json(x):
        # Recursively convert objects into JSON-serializable forms (fallback to str)
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, dict):
            return {str(k): _sanitize_for_json(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize_for_json(v) for v in x]
        try:
            # Try to serialize common types directly
            return json.loads(json.dumps(x))
        except Exception:
            try:
                return str(x)
            except Exception:
                return "<unserializable>"

    final_text = final_msg.content or ""

    # Export conversation (downloadable)
    convo_path = export_agent_conversation(messages, tool_calls_info, tool_results_dict, final_text)

    result_obj = {
        "tool_calls": tool_calls_info,
        "tool_results": tool_results_dict,
        "final_response": final_text,
        "conversation_export_path": convo_path,
    }

    if not final_text.strip():
        # Add readable debug information
        debug = {
            "messages": _sanitize_for_json(messages),
            "assistant_msg_tool_calls": _sanitize_for_json(getattr(choice, 'message', None) and getattr(choice.message, 'tool_calls', None)),
            "first_choice": _sanitize_for_json(choice),
            "second_choices": _sanitize_for_json(getattr(second, 'choices', None)),
            "tool_results": _sanitize_for_json(tool_results_dict),
        }
        result_obj["debug"] = debug

    return result_obj

