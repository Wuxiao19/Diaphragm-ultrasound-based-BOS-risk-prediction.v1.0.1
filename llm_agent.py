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
# BOS literature knowledge loader
# ============================================================

_PAPER_DIR = Path(__file__).resolve().parent / "paper"

_PAPER_META = [
    {
        "file": "BOS diagnosis-NIH.pdf",
        "label": "NIH 2014 Chronic GVHD Diagnosis & Staging Consensus",
        "key_sections": ["diagnosis", "staging", "BOS criteria", "lung scoring"],
    },
    {
        "file": "BOS中国指南.pdf",
        "label": "Chinese Expert Consensus on BOS Diagnosis & Treatment (2022)",
        "key_sections": ["diagnostic criteria", "grading", "treatment", "monitoring", "prevention"],
    },
    {
        "file": "Eur Respir J-2024-Bos-Treatment.pdf",
        "label": "ERS/EBMT 2024 Clinical Practice Guidelines on Treatment of Pulmonary cGvHD-BOS",
        "key_sections": ["ICS/LABA", "FAM therapy", "ibrutinib", "ruxolitinib", "lung transplant", "follow-up"],
    },
]

_bos_knowledge_cache: Optional[str] = None

def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    from pypdf import PdfReader
    print(f"loading pypdf success", file=sys.stderr, flush=True)
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise RuntimeError(
            f"No text extracted from PDF: {pdf_path}"
        )
    return text

def load_bos_knowledge(max_chars_per_paper: int = 6000) -> str:
    """Load and cache BOS literature knowledge from the paper/ directory."""
    global _bos_knowledge_cache
    if _bos_knowledge_cache is not None:
        return _bos_knowledge_cache

    blocks: List[str] = []
    for meta in _PAPER_META:
        pdf_path = _PAPER_DIR / meta["file"]
        if not pdf_path.exists():
            continue
        raw_text = _extract_pdf_text(pdf_path)
        if not raw_text.strip():
            continue
        # Trim to a reasonable length to avoid bloating the prompt
        trimmed = raw_text[:max_chars_per_paper]
        block = (
            f"### Reference: {meta['label']}\n"
            f"Key topics covered: {', '.join(meta['key_sections'])}\n\n"
            f"{trimmed}"
        )
        blocks.append(block)

    if blocks:
        _bos_knowledge_cache = (
            "=== BOS Clinical Literature Knowledge Base ===\n"
            "The following excerpts are drawn from peer-reviewed guidelines and consensus documents "
            "on Bronchiolitis Obliterans Syndrome (BOS) after allogeneic hematopoietic stem cell "
            "transplantation (allo-HSCT). Use this knowledge to support your clinical interpretations "
            "and suggestions. Always remind the user that model predictions cannot replace formal "
            "clinical evaluation.\n\n"
            + "\n\n---\n\n".join(blocks)
        )
    else:
        _bos_knowledge_cache = ""

    return _bos_knowledge_cache


# ============================================================
# Environment variables & LLM client initialization
# ============================================================

# DEFAULT_LLM_BASE_URL = "https://api.aipaibox.com/v1"
# DEFAULT_LLM_MODEL = "gpt-5.4"

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


async def get_mcp_client(mcp_entry: str = "mcp_tools.py") -> Client:
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


async def mcp_list_tools(mcp_entry: str = "mcp_tools.py") -> List[Dict[str, Any]]:
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
- Looks like "YY-MM-DD-A123", "YY-MM-DD-B123", or "YY-MM-DD-C123".
- "YY-MM-DD" is the exam date (20YY-MM-DD).
- "A123/B123/C123" is the patient ID.

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

Additional optional clinical factors may be provided separately, including Sex, Age, BMI, Complication, cGVHD, and Time-HSCT.
These factors are reference-only context for interpretation and suggestions.
They do not change tool selection, image-based prediction, or predicted probability.
If such factors are provided, incorporate them cautiously into the high-risk patient analysis, high-risk patient suggestions,
recheck patient analysis, and recheck patient suggestions when relevant.
If they are not provided, do not speculate.

When generating clinical suggestions, you may draw on the BOS literature knowledge base provided below (if any).
Cite the specific guideline or consensus document by name (e.g., "per the Chinese Expert Consensus 2022" or
"per the ERS/EBMT 2024 guidelines") when the suggestion is directly supported by that source.
Do not fabricate citations or invent guideline content.
"""

def _build_reference_context_message(
    single_reference_factors: Optional[Dict[str, Any]] = None,
    batch_reference_records: Optional[List[Dict[str, Any]]] = None,
    batch_reference_filename: Optional[str] = None,
) -> Optional[str]:
    """Build a prompt block for optional clinical reference factors."""
    if single_reference_factors:
        clean_single = {
            k: v for k, v in single_reference_factors.items()
            if v not in (None, "", [])
        }
        if clean_single:
            return (
                "The following optional clinical reference factors were provided for this single case. "
                "Use them only as supportive context within the high-risk or recheck-related analysis and suggestions when relevant. "
                "Do not treat them as model inputs and do not alter the image-based prediction:\n\n"
                f"{json.dumps(clean_single, ensure_ascii=False, indent=2)}"
            )

    if batch_reference_records:
        clean_records = []
        for row in batch_reference_records:
            if isinstance(row, dict):
                clean_row = {k: v for k, v in row.items() if v not in (None, "", [])}
                if clean_row:
                    clean_records.append(clean_row)
        if clean_records:
            payload = {"source_file": batch_reference_filename,"records": clean_records,}
            return (
                "An optional batch clinical reference table was uploaded for interpretation only. "
                "Use it only when a record can be matched to a case by merged_key or patient/date fields, "
                "and integrate it into high-risk or recheck-related analysis and suggestions when relevant. "
                "Do not change the prediction outputs based on this table:\n\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )

    return None


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
        client = await get_mcp_client("mcp_tools.py")
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
    async with Client(_resolve_mcp_entry("mcp_tools.py")) as transient_client:
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
    single_reference_factors: Optional[Dict[str, Any]] = None,
    batch_reference_records: Optional[List[Dict[str, Any]]] = None,
    batch_reference_filename: str = None,
) -> Dict[str, Any]:

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

    # Build system prompt with optional BOS literature knowledge
    bos_knowledge = load_bos_knowledge()
    if bos_knowledge:
        system_content = (
            LLM_SYSTEM_PROMPT.rstrip()
            + "\n\n"
            + bos_knowledge
        )
    else:
        system_content = LLM_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]

    reference_context_message = _build_reference_context_message(
        single_reference_factors=single_reference_factors,
        batch_reference_records=batch_reference_records,
        batch_reference_filename=batch_reference_filename,
    )
    if reference_context_message:
        messages.append({"role": "user", "content": reference_context_message})

    # Try to fetch tools dynamically from MCP
    tools = await mcp_list_tools("mcp_tools.py")

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
                    " and missing modality statistics. If optional clinical reference factors"
                    " were provided earlier, use them only as supplementary interpretation"
                    " context for the matched case(s):\n\n"
                    f"{summary_str}\n\n"
                    "When producing the final English report, please:\n"
                    "1) Correctly interpret the exam date and patient ID in merged_key;\n"
                    "2) Highlight patients with rechecks and compare risk trends across dates;\n"
                    "3) If missing modality cases exist, mention them explicitly;\n"
                    "4) Use optional clinical reference factors only if they are present and match a case;\n"
                    "5) Do not create a separate section for clinical factors; instead blend them into the high-risk or recheck-related analysis and suggestions;\n"
                    "6) Follow the remaining system instructions as stated."
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
