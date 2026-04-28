"""Tools available to the ReAct agent.

Tools access the Memory instance through a global singleton set at bot
startup (set_memory()). This avoids passing dynamic state through the
ReAct signatures, which don't handle runtime parameters well.
"""
import subprocess
import os
from datetime import datetime
from typing import Optional, Any
from langchain_core.tools import tool
from telegram import Bot

from src.memory import Memory


_memory: Optional[Memory] = None
_bot: Optional[Bot] = None
_chat_id: Optional[int] = None


def set_memory(m: Memory) -> None:
    global _memory
    _memory = m


def set_bot_context(bot: Bot, chat_id: int) -> None:
    global _bot, _chat_id
    _bot = bot
    _chat_id = chat_id


def _require_memory() -> Memory:
    if _memory is None:
        raise RuntimeError("Memory not initialized. Call set_memory() at startup.")
    return _memory


def _require_bot():
    if _bot is None or _chat_id is None:
        raise RuntimeError("Bot context not initialized. Call set_bot_context() at startup.")
    return _bot, _chat_id


@tool
def get_current_datetime() -> str:
    """Return the current date and time in a human-readable format.
    Use ONLY if the user asks what day/time it is, or if you need to
    propose a specific time for an appointment.
    """
    now = datetime.now()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    weekday = days[now.weekday()]
    month = months[now.month - 1]
    return f"{weekday} {now.day} {month} {now.year}, {now.hour:02d}:{now.minute:02d}"


@tool
def remember_this(fact: str) -> str:
    """Save an important fact about the user you're chatting with.
    Use when he shares long-term info: name, job, strong preferences
    (favorite music/film/food), important people, dates, places.
    DO NOT save trivia or small talk.
    Write the fact as a short, clear sentence.
    Example: "His name is Matt, lives in Milan, works in AI".
    """
    mem = _require_memory()
    mem.save_fact(fact.strip())
    return f"ok, remembered: {fact.strip()}"


@tool
def execute_shell(command: str) -> str:
    """Execute a shell command in the local environment and return output.
    USE EXTREME CAUTION. Only use when asked to run code, check system,
    or perform technical tasks.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        out = result.stdout or ""
        err = result.stderr or ""
        return f"STDOUT:\n{out}\nSTDERR:\n{err}" if (out or err) else "Command finished (no output)"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def list_files(path: str = ".") -> str:
    """List files in a directory."""
    try:
        files = os.listdir(path)
        return "\n".join(files) if files else "Directory is empty"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def read_file(file_path: str) -> str:
    """Read content of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write or overwrite a file with content."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully written to {file_path}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
async def send_telegram_file(file_path: str, caption: str = "") -> str:
    """Send a file (document) to the user via Telegram. 
    Use this to share generated scripts, data, or any local files.
    """
    bot, chat_id = _require_bot()
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."
    
    try:
        with open(file_path, "rb") as f:
            await bot.send_document(chat_id=chat_id, document=f, caption=caption)
        return f"Successfully sent {file_path} to the user."
    except Exception as e:
        return f"Error sending file: {str(e)}"


@tool
async def generate_and_send_photo(prompt: str) -> str:
    """Generate an image using AI and send it to the user immediately.
    The prompt should be in English, describing the scene (e.g., 'wearing a red dress, in a cafe').
    This is the best way to show yourself or any scene.
    """
    bot, chat_id = _require_bot()
    from src.pic_flow import run_pic_flow, can_send_pic
    from src.persona import ensure_today_mood
    from src.memory import AsyncMemory
    
    amem = AsyncMemory(_require_memory())
    ok, reason = await can_send_pic(amem)
    if not ok:
        return f"Cannot generate photo right now: {reason}"
    
    mood = await ensure_today_mood(amem)
    
    try:
        # We use 'manual' trigger type for tool-based generation
        caption = await run_pic_flow(
            bot=bot,
            amem=amem,
            chat_id=chat_id,
            trigger_type="manual",
            explicit_prompt=prompt,
            mood=mood
        )
        if caption:
            return f"Photo successfully generated and sent with caption: {caption}"
        else:
            return "Photo generation failed or was cancelled."
    except Exception as e:
        return f"Error in photo generation tool: {str(e)}"


ALL_TOOLS = [
    get_current_datetime, 
    remember_this, 
    execute_shell, 
    list_files, 
    read_file, 
    write_file,
    send_telegram_file,
    generate_and_send_photo
]
