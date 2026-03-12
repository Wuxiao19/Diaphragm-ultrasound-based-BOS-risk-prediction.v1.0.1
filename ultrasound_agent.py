"""
Agent construction using an LLM + FastMCP + MCP tools .

- run_llm_agent: let the LLM decide which MCP tool to call and then summarize the result.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastmcp import Client
from openai import OpenAI
import shutil
import time

# ============================================================
# Environment variables & LLM client initialization
# ============================================================

DEFAULT_LLM_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B"


def get_llm_client(api_key: str, base_url: str = DEFAULT_LLM_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)

# ============================================================
# MCP client reuse & tool discovery
# ============================================================
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


async def mcp_list_tools(mcp_entry: str = "agent_et_mcp.py") -> List[Dict[str, Any]]:
    """
    Obtain tool definitions from the MCP server.
    """
    client = await get_mcp_client(mcp_entry)
    tools = await client.list_tools()

    llm_tools = []

    for t in tools:
        llm_tools.append({
            "type": "function",
            "function": {"name": t.name,"description": getattr(t, "description", ""),
                "parameters": getattr(t,"inputSchema",{"type": "object","properties": {}},),},
        })

    return llm_tools   

# ============================================================
# LLM explanation section
# ============================================================

LLM_SYSTEM_PROMPT = """
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



def _track_detect_output_dir(normalized: Dict[str, Any]) -> None:
    """Track detect_output_dir for cleanup."""
    try:
        d = normalized.get("detect_output_dir")
        if isinstance(d, str) and d and d not in _last_detect_output_dirs:
            _last_detect_output_dirs.append(d)
    except Exception:
        pass


async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    # First try the reused global client (performance; avoid frequent restarts)
    try:
        client = await get_mcp_client("agent_et_mcp.py")
        try:
            result = await client.call_tool(tool_name, arguments)
        except RuntimeError:
            client = None
        else:
            normalized = _normalize_mcp_result(result)
            _track_detect_output_dir(normalized)
            return normalized
    except Exception:
        client = None

    # If the reused client is unavailable, use a short-lived async context client
    async with Client(_resolve_mcp_entry("agent_et_mcp.py")) as transient_client:
        result = await transient_client.call_tool(tool_name, arguments)
        normalized = _normalize_mcp_result(result)
        _track_detect_output_dir(normalized)
        return normalized


async def _cleanup_previous_detect_outputs() -> None:
    """Delete detect_output_dir recorded from the previous run (if any) and clear the list."""
    global _last_detect_output_dirs
    if not _last_detect_output_dirs:
        return
    for d in list(_last_detect_output_dirs):
        try:
            if d and os.path.exists(d) and os.path.isdir(d) and d.startswith("detect/"):
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
    Normalize MCP tool results into a Python dict.

    Expected MCP structure:
        CallToolResult
            └ content[0].text -> JSON string
    """

    # 1. Already dict
    if isinstance(result, dict):
        return result

    # 2. MCP CallToolResult.content
    content = getattr(result, "content", None)
    if content and len(content) > 0:
        first = content[0]
        text = getattr(first, "text", None)

        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return {"raw": text}

    # 3. FastMCP sometimes returns list
    if isinstance(result, list) and result:
        first = result[0]
        text = getattr(first, "text", None)

        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return {"raw": text}

    # 4. Fallback
    return {"raw": str(result)}


async def run_llm_agent(
    b_image_path: str = None,
    m_image_path: str = None,
    b_folder_path: str = None,
    m_folder_path: str = None,
    api_key: str = None,
    base_url: str = DEFAULT_LLM_BASE_URL,
    model: str = DEFAULT_LLM_MODEL,
    user_query: str = None,
) -> Dict[str, Any]:
    """
    Args:
        - b_image_path, m_image_path: image paths for a single exam
        - b_folder_path, m_folder_path: folder paths for batch exams
        - api_key: LLM API key
        - base_url: API base URL (default SiliconFlow)
        - model: model name (default Qwen3-8B)
        - user_query: user natural language request (auto-generated)

    Returns:
        {
            "tool_calls": [...],  # tool call records
            "tool_results": {...},  # tool results
            "final_response": "...",  # LLM final response
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
2. Choose and call the correct batch detection tool.
3. After getting the JSON result, summarize the risk distribution across patients.
4. Identify high-risk patients (risk_probability > 0.6) and list their IDs and probabilities.
5. Provide 1–3 clinical or management suggestions.
6. Remind that this is a model prediction and cannot replace a doctor's diagnosis.
"""
        else:
            raise ValueError("Provide (b_image_path, m_image_path) or (b_folder_path, m_folder_path)")

    # Clean up outputs from the previous run (if any)
    await _cleanup_previous_detect_outputs()

    # Get LLM client
    client = get_llm_client(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    # Try to fetch tools dynamically from MCP
    tools = await mcp_list_tools("agent_et_mcp.py")

    # Step 1: let the LLM decide which tool to call (dynamic tools list)
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

        # Try to read structured summary from tool result (only once)
        if detection_summary is None and isinstance(tool_result, dict):
            summary = tool_result.get("detection_summary")
            if isinstance(summary, dict):
                detection_summary = summary

    # If a structured summary is available, provide it to the LLM as extra context,
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

    # Step 3: let the LLM produce the final response based on tool results and summary
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

    raw_final_text = final_msg.content or ""
    final_text = raw_final_text if raw_final_text.strip() else "Please try again."

    # Export conversation (downloadable)
    convo_path = export_agent_conversation(messages, tool_calls_info, tool_results_dict, final_text)

    result_obj = {
        "tool_calls": tool_calls_info,
        "tool_results": tool_results_dict,
        "final_response": final_text,
        "conversation_export_path": convo_path,
    }

    if not raw_final_text.strip():
        print("=" * 60, file=sys.stderr, flush=True)
        print("[Agent Debug] LLM returned empty response", file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)
        debug = {
            "messages": _sanitize_for_json(messages),
            "assistant_msg_tool_calls": _sanitize_for_json(getattr(choice, 'message', None) and getattr(choice.message, 'tool_calls', None)),
            "first_choice": _sanitize_for_json(choice),
            "second_choices": _sanitize_for_json(getattr(second, 'choices', None)),
            "tool_results": _sanitize_for_json(tool_results_dict),
        }
        result_obj["debug"] = debug
        print(json.dumps(debug, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)

    return result_obj

