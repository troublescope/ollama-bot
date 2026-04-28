"""ReAct agent with memory (history + facts) injected into the context.

Exposes:
  - `generate_reply`: ReAct-based text reply
  - `generate_vision_reply`: direct LLM reply for photo comments (no ReAct)
  - `extract_facts`: second-pass LLM to pull personal facts from user text
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import create_react_agent

from src.config import CONFIG
from src.tools import ALL_TOOLS
from src.persona import get_persona_system_prompt


log = logging.getLogger("diana-bot.agent")
TOOLS_NOTE = """\n\nAVAILABLE TOOLS:
- get_current_datetime: only if asked for date/time.
- remember_this: save important long-term facts about the user.
- execute_shell: run terminal commands (linux/bash).
- list_files, read_file, write_file: manage local project files.
- send_telegram_file: send any local file/document to the user via Telegram.
- generate_and_send_photo: use AI to generate and send a photo to the user.

TECHNICAL/CODING MODE:
...
"""

If the user asks for coding, debugging, or system tasks, you switch to 'Expert Dev Mode'. 
Maintain your personality but prioritize technical accuracy and successful execution. 
You can read files to understand the project, write code, and run it to verify results.
"""


_agent_flash: Any = None
_agent_pro: Any = None
_llm_vision: Any = None


def _create_llm(model_name: str | None = None, **kwargs) -> Any:
    """Helper to create an LLM instance. Uses ChatOpenAI for cloud providers
    (like Gemini on Ollama Cloud) because it handles thought signatures 
    and tool calling much more reliably than ChatOllama.
    """
    host = CONFIG.ollama_host.lower()
    is_cloud = "ollama.com" in host or ("localhost" not in host and "127.0.0.1" not in host)
    
    target_model = model_name or CONFIG.ollama_model
    
    if is_cloud:
        # For OpenAI-compatible cloud endpoints
        api_base = CONFIG.ollama_host
        if not api_base.endswith("/v1") and not api_base.endswith("/v1/"):
            api_base = f"{api_base.rstrip('/')}/v1"
            
        # ChatOpenAI uses 'max_tokens' instead of 'num_predict'
        # and doesn't support 'reasoning', 'keep_alive', 'num_ctx'
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ["reasoning", "keep_alive", "num_predict", "num_ctx"]}
        if "num_predict" in kwargs:
            openai_kwargs["max_tokens"] = kwargs["num_predict"]
            
        log.info("Using ChatOpenAI (cloud mode) base: %s, model: %s", api_base, target_model)

        return ChatOpenAI(
            model=target_model,
            openai_api_base=api_base,
            openai_api_key=CONFIG.ollama_api_key or "no-key",
            **openai_kwargs
        )

    log.info("Using ChatOllama (local mode) model: %s", target_model)
    client_kwargs = {}
    if CONFIG.ollama_api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {CONFIG.ollama_api_key}"}

    return ChatOllama(
        model=target_model,
        base_url=CONFIG.ollama_host,
        client_kwargs=client_kwargs,
        **kwargs
    )


def get_agent(use_pro: bool = False) -> Any:
    global _agent_flash, _agent_pro
    
    if use_pro:
        if _agent_pro is None:
            llm = _create_llm(
                model_name=CONFIG.ollama_model_pro,
                reasoning=True,
                temperature=0.7,
                num_ctx=16384,
                num_predict=2048,
                keep_alive=CONFIG.ollama_keep_alive,
                streaming=True,
            )
            _agent_pro = create_react_agent(model=llm, tools=ALL_TOOLS)
            log.info("Pro Agent ready: model=%s", CONFIG.ollama_model_pro)
        return _agent_pro
    else:
        if _agent_flash is None:
            llm = _create_llm(
                model_name=CONFIG.ollama_model,
                reasoning=True,
                temperature=0.8,
                num_ctx=8192,
                num_predict=1024,
                keep_alive=CONFIG.ollama_keep_alive,
                streaming=True,
            )
            _agent_flash = create_react_agent(model=llm, tools=ALL_TOOLS)
            log.info("Flash Agent ready: model=%s", CONFIG.ollama_model)
        return _agent_flash


def get_vision_llm() -> Any:
    """Direct LLM (no ReAct) for the vision path: avoids the doubled latency
    that ReAct would add on top of an already heavy image payload.
    """
    global _llm_vision
    if _llm_vision is None:
        _llm_vision = _create_llm(
            reasoning=True,
            temperature=0.8,
            num_ctx=8192,
            num_predict=1024,
            keep_alive=CONFIG.ollama_keep_alive,
            streaming=True,
        )
        log.info("Vision LLM ready (direct, no ReAct)")
    return _llm_vision


def _format_event_line(ev: dict) -> str:
    ts = ev.get("created_at", "")
    # created_at is ISO "YYYY-MM-DDTHH:MM:SS"; render short month+day+time
    try:
        from datetime import datetime as _dt
        d = _dt.fromisoformat(ts)
        label = d.strftime("%d %b %H:%M")
    except Exception:
        label = ts[:16]
    return f"[{label}] {ev.get('text','').strip()}"


def _build_system_prompt(
    facts: list[str],
    summary: str | None,
    with_tools: bool = False,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
) -> str:
    parts = [
        get_persona_system_prompt(
            daily_mood=daily_mood,
            intimacy_count=intimacy_count,
        )
    ]
    if with_tools:
        parts.append(TOOLS_NOTE)
    if facts:
        parts.append(
            "\nTHINGS YOU KNOW ABOUT HIM (always true):\n- "
            + "\n- ".join(facts)
        )
    if events:
        lines = [_format_event_line(e) for e in events]
        parts.append(
            "\nRECENT EVENTS IN HIS LIFE (told outside chat in the last days):\n- "
            + "\n- ".join(lines)
        )
    if summary:
        parts.append(
            "\nSUMMARY OF PREVIOUS CONVERSATIONS:\n" + summary
        )
    return "\n".join(parts)


def _history_to_messages(history: list[dict]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for row in history:
        if row["role"] == "user":
            out.append(HumanMessage(content=row["content"]))
        elif row["role"] == "assistant":
            out.append(AIMessage(content=row["content"]))
    return out


async def generate_reply(
    user_text: str,
    history: list[dict] | None = None,
    facts: list[str] | None = None,
    summary: str | None = None,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
) -> str:
    """Text reply via ReAct agent (tool calling + memory)."""
    agent = get_agent()
    system = _build_system_prompt(
        facts or [], summary, with_tools=True,
        daily_mood=daily_mood, intimacy_count=intimacy_count,
        events=events,
    )
    msgs: list[BaseMessage] = [SystemMessage(content=system)]
    msgs.extend(_history_to_messages(history or []))
    msgs.append(HumanMessage(content=user_text))

    log.debug("Agent request: msgs=%d user=%r", len(msgs), user_text)
    result = await agent.ainvoke({"messages": msgs})
    messages = result.get("messages", [])

    final_text = ""
    tool_calls_used: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", []) or []:
                tool_calls_used.append(tc.get("name", "?"))
            content = m.content if isinstance(m.content, str) else ""
            if content.strip():
                final_text = content.strip()
    if tool_calls_used:
        log.info("Agent used tools: %s", tool_calls_used)
    log.info("Agent reply (%d chars): %s", len(final_text), final_text[:120])
    return final_text or "..."


async def stream_reply(
    user_text: str,
    history: list[dict] | None = None,
    facts: list[str] | None = None,
    summary: str | None = None,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
):
    """Yield text chunks via ReAct agent streaming. 
    Switches to Pro model if technical task is detected.
    """
    # Simple intent detection for Pro model
    tech_keywords = [
        "code", "coding", "script", "python", "bash", "shell", "terminal", 
        "file", "folder", "directory", "debug", "error", "fix", "ram", "cpu", 
        "disk", "system", "install", "git", "buatkan", "bikin", "tulis"
    ]
    is_tech = any(kw in user_text.lower() for kw in tech_keywords)
    
    agent = get_agent(use_pro=is_tech)
    
    system = _build_system_prompt(
        facts or [], summary, with_tools=True,
        daily_mood=daily_mood, intimacy_count=intimacy_count,
        events=events,
    )
    msgs: list[BaseMessage] = [SystemMessage(content=system)]
    msgs.extend(_history_to_messages(history or []))
    msgs.append(HumanMessage(content=user_text))

    log.debug("Agent stream request (%s): msgs=%d user=%r", 
              "PRO" if is_tech else "FLASH", len(msgs), user_text)

    async for chunk in agent.astream({"messages": msgs}, stream_mode="messages"):
        msg, metadata = chunk
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            if isinstance(msg.content, str) and msg.content:
                yield msg.content


FACT_EXTRACTION_SYSTEM = """You are a fact extractor. Analyze the user's message \
and identify ONLY important long-term personal information about the sender:
- name, age, job, where they live
- important people (children, partner, friends), pets, relationships
- strong durable preferences (music, food, sports, favorite books)
- important dates, anniversaries
- life projects, milestones, significant events

IGNORE small talk, passing comments, fleeting opinions, questions, greetings, mood-of-the-moment.

Reply with ONE FACT PER LINE, short and clear, in third person singular \
("His name is Matt", "He's 34", "He has a son", "He works in AI").

If the message contains NO relevant personal facts, reply exactly with: NONE

No other explanation, no bullet points, just the list or NONE."""

_llm_extract: Any = None


def _get_extract_llm() -> Any:
    """Minimal dedicated LLM for fact extraction (low num_predict for speed)."""
    global _llm_extract
    if _llm_extract is None:
        _llm_extract = _create_llm(
            reasoning=True,
            temperature=0.2,
            num_ctx=2048,
            num_predict=120,
            keep_alive=CONFIG.ollama_keep_alive,
        )
    return _llm_extract



async def extract_facts(user_text: str, timeout_s: float = 90.0) -> list[str]:
    """Minimal LLM pass that extracts personal facts from a message.
    Internal timeout: if the LLM takes longer than `timeout_s`, returns [].
    """
    if len(user_text.strip()) < 15:
        return []
    llm = _get_extract_llm()
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=FACT_EXTRACTION_SYSTEM),
                HumanMessage(content=user_text),
            ]),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        log.warning("fact extraction timeout after %.0fs (msg: %r)", timeout_s, user_text[:60])
        return []
    except Exception:
        log.exception("fact extraction LLM call failed")
        return []
    text = (resp.content if isinstance(resp.content, str) else "").strip()
    if not text or text.upper().startswith("NONE"):
        return []
    # Split by line and clean up.
    facts: list[str] = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-*•").strip()
        if not line or line.upper() == "NONE":
            continue
        if len(line) < 5 or len(line) > 200:
            continue
        facts.append(line)
    return facts


async def generate_vision_reply(
    caption: str,
    image_b64: str,
    facts: list[str] | None = None,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
) -> str:
    """Photo reply: direct path (no ReAct) to keep latency manageable."""
    llm = get_vision_llm()
    system = _build_system_prompt(
        facts or [], summary=None,
        daily_mood=daily_mood, intimacy_count=intimacy_count,
        events=events,
    )

    text_prompt = caption.strip() if caption else "sending you a photo, tell me what you think"
    msgs: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(
            content=[
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ]
        ),
    ]
    log.debug("Vision request: caption=%r img_b64_len=%d", caption, len(image_b64))
    resp = await llm.ainvoke(msgs)
    text = (resp.content if isinstance(resp.content, str) else "").strip()
    log.info("Vision reply (%d chars): %s", len(text), text[:120])
    return text or "..."


async def stream_vision_reply(
    caption: str,
    image_b64: str,
    facts: list[str] | None = None,
    daily_mood: dict | None = None,
    intimacy_count: int | None = None,
    events: list[dict] | None = None,
):
    """Yield chunks for vision reply."""
    llm = get_vision_llm()
    system = _build_system_prompt(
        facts or [], summary=None,
        daily_mood=daily_mood, intimacy_count=intimacy_count,
        events=events,
    )

    text_prompt = caption.strip() if caption else "sending you a photo, tell me what you think"
    msgs: list[BaseMessage] = [
        SystemMessage(content=system),
        HumanMessage(
            content=[
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ]
        ),
    ]
    log.debug("Vision stream request: caption=%r", caption)
    async for chunk in llm.astream(msgs):
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content
