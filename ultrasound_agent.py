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
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI


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
    async with Client(mcp_entry) as client:
        result = await client.call_tool(
            "detect_single_pair",
            {
                "b_image_path": b_image_path,
                "m_image_path": m_image_path,
            },
        )
        # FastMCP 一般返回 Content 列表，这里尽量解析成 dict
        if isinstance(result, list) and result:
            first = result[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}


async def mcp_detect_batch_folders(
    b_folder_path: str,
    m_folder_path: str,
    mcp_entry: str = "agent_et_mcp.py",
) -> Dict[str, Any]:
    """
    通过 FastMCP Client 调用 MCP 工具 detect_batch_folders，获取批量 B/M 检测结果。
    """
    async with Client(mcp_entry) as client:
        result = await client.call_tool(
            "detect_batch_folders",
            {
                "b_folder_path": b_folder_path,
                "m_folder_path": m_folder_path,
            },
        )
        if isinstance(result, list) and result:
            first = result[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}


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

当你已经拿到 JSON 结构的“自动检测结果”时，其典型内容包括：
- 对于单个患者：merged_key、risk_probability、prediction、prediction_label 等字段；
- 对于多个患者：上述字段的列表、总样本数、平均风险等信息。

你的任务：
1. 用清晰、专业但尽量易懂的中文解释这些检测结果的含义；
2. 根据 risk_probability 对风险进行分级（如很低、较低、中等、较高、很高）；
3. 如果是批量结果，请指出其中明显高风险的患者（例如风险概率 > 0.7），并列出他们的 ID 和概率；
4. 给出 1~3 条临床或管理上的建议（例如是否需要进一步检查、随访、注意点等）；
5. 明确提醒：这是基于图像的机器学习模型结果，不能替代医生的最终诊断。
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
# 示例 1：单组 B/M + Qwen 解释（你之前已经验证通过的简单链路）
# ============================================================


async def run_single_example() -> None:
    """
    示例：一组 B/M 图片 → MCP 检测 → Qwen 解释。

    你可以先用你之前已经跑通的那组图片路径，确认整个链路没问题。
    """
    # 你可以把这里换成你自己的图片路径
    b_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\B_model_one\25-08-11-B013_25-08-11-B013-L-Tdi-exp1.jpg"
    m_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\M_model_one\25-08-11-B013_25-08-11-B013-R-DE-DB1.jpg"

    if not Path(b_image_path).exists() or not Path(m_image_path).exists():
        print("❌ 示例 B/M 图片路径不存在，请先修改 deepseek_ultrasound_agent.py 中的路径。")
        return

    print("==============================================")
    print("  步骤1：通过 MCP 工具进行单组 B/M 检测")
    print("==============================================")
    detection = await mcp_detect_single_pair(b_image_path, m_image_path)
    print("原始检测结果（截断展示）：")
    print(json.dumps(detection, ensure_ascii=False, indent=2)[:600] + "...\n")

    print("==============================================")
    print("  步骤2：调用 Qwen 模型做中文解释")
    print("==============================================\n")

    user_intent = (
        "请根据这名患者的检测结果，说明大致患病风险概率和风险级别，并给出 1~3 条临床建议。"
    )
    # 说明：示例里为了方便，你可以直接在这里临时填 key；
    # 更推荐的做法：用环境变量/`.env`/Streamlit 输入框传入。
    # api_key = os.getenv("QWEN_API_KEY", "").strip()
    api_key = "sk-jxrjjptjthxipidmqageemugyjpsqvzxdabusdngydpvsxxf"
    if not api_key:
        print("❌ 未找到 QWEN_API_KEY（环境变量或 .env），无法调用 Qwen。")
        return

    explanation = await qwen_explain_detection(
        detection_json=detection,
        user_intent=user_intent,
        api_key=api_key,
    )

    print("******** Qwen 模型的中文解读 ********\n")
    print(explanation)
    print("\n****************************************\n")


# ============================================================
# 示例 2：让 Qwen 自己决定何时、如何调用 MCP 工具（Agent 行为）
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


async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    由 Agent 示例使用：根据工具名和参数真正调用 MCP 工具。
    这里直接使用 FastMCP Client 的通用调用方式。
    """
    async with Client("agent_et_mcp.py") as client:
        result = await client.call_tool(tool_name, arguments)
        if isinstance(result, list) and result:
            first = result[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
        if isinstance(result, dict):
            return result
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
    真正的 Agent 行为——让 Qwen 自己“决定调用哪个检测工具”。

    参数：
      - b_image_path, m_image_path: 单组检测的图片路径
      - b_folder_path, m_folder_path: 批量检测的文件夹路径
      - api_key: Qwen API Key
      - base_url: API base URL（默认 SiliconFlow）
      - model: 模型名称（默认 Qwen3-8B）
      - user_query: 用户自然语言需求（可选，会自动生成）

    返回：
      {
        "tool_calls": [...],  # 工具调用记录
        "tool_results": {...},  # 工具返回结果
        "final_response": "...",  # Qwen 最终回答
      }
    """
    if not api_key:
        raise ValueError("必须提供 api_key")

    # 自动生成 user_query（如果未提供）
    if not user_query:
        if b_image_path and m_image_path:
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

    client = get_qwen_client(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    # 步骤1：让 Qwen 决定调用哪个工具
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=QWEN_TOOLS,
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
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

    # 步骤3：让 Qwen 基于工具结果给出最终总结
    second = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    final_msg = second.choices[0].message

    return {
        "tool_calls": tool_calls_info,
        "tool_results": tool_results_dict,
        "final_response": final_msg.content or "",
    }


async def run_agent_example() -> None:
    """
    命令行示例：运行 Qwen Agent。
    """
    b_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\B_model_one\25-08-11-B013_25-08-11-B013-L-Tdi-exp1.jpg"
    m_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\M_model_one\25-08-11-B013_25-08-11-B013-R-DE-DB1.jpg"

    if not Path(b_image_path).exists() or not Path(m_image_path).exists():
        print("❌ 示例 B/M 图片路径不存在，请先修改 deepseek_ultrasound_agent.py 中的路径。")
        return

    api_key = "sk-jxyjpsqvzxdabusdngydpvsxxf"

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
    # 1) 只看 JSON + 解释（你之前已经测试通过的）
    # await run_single_example()

    # 2) 让 Qwen 亲自调用工具（真正 Agent 行为）
    await run_agent_example()


if __name__ == "__main__":
    asyncio.run(main())


