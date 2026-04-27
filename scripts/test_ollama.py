"""Quick smoke test of the Ollama model - text + image."""
import sys
import time
import base64
from pathlib import Path

import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")


def create_llm(**kwargs) -> ChatOllama:
    client_kwargs = {}
    if OLLAMA_API_KEY:
        client_kwargs["headers"] = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        client_kwargs=client_kwargs,
        **kwargs
    )


def test_text() -> None:
    print(f"\n[TEST 1] Text with {OLLAMA_MODEL} at {OLLAMA_HOST}")
    llm = create_llm(temperature=0.7)
    messages = [
        SystemMessage(content="You are a friendly 24-year-old. Reply short, informal, in English."),
        HumanMessage(content="Hi, how's it going?"),
    ]
    start = time.time()
    resp = llm.invoke(messages)
    elapsed = time.time() - start
    print(f"  Reply: {resp.content}")
    print(f"  Time: {elapsed:.2f}s")


def test_vision(image_path: str | None = None) -> None:
    print(f"\n[TEST 2] Image with {OLLAMA_MODEL} at {OLLAMA_HOST}")
    if image_path is None or not Path(image_path).exists():
        print("  (skipping: no test image - pass a path as argv[1] to test)")
        return

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    llm = create_llm(temperature=0.7)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "What do you see in this photo? Reply short and natural."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"},
        ]
    )
    start = time.time()
    resp = llm.invoke([msg])
    elapsed = time.time() - start
    print(f"  Reply: {resp.content}")
    print(f"  Time: {elapsed:.2f}s")


if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_text()
    test_vision(image_arg)
    print("\nTest complete.")
