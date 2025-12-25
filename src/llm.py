# src/llm.py
import os
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_ollama_model() -> str:
    """Cache the model name to avoid repeated env lookups."""
    return os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def call_ollama(prompt: str, model: str | None = None, timeout: int = 300) -> str:
    """
    Calls Ollama via CLI safely on Windows by sending UTF-8 bytes to stdin.
    This avoids UnicodeEncodeError (cp1252/charmap).
    
    Performance tips:
    - Ollama keeps model loaded for ~5 min after first call (warm cache)
    - First call may be slow due to model loading
    - Subsequent calls are much faster
    """
    if model is None:
        model = _get_ollama_model()

    # Ensure prompt is always str
    prompt = "" if prompt is None else str(prompt)

    try:
        # Send UTF-8 bytes (NOT text=True)
        # Using --nowordwrap to avoid formatting overhead
        result = subprocess.run(
            ["ollama", "run", model, "--nowordwrap"],
            input=prompt.encode("utf-8", errors="replace"),
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return "❌ Ollama not found. Make sure Ollama is installed and `ollama` is in your PATH."
    except subprocess.TimeoutExpired:
        return "⏳ Ollama timed out. Try a smaller model or reduce context length."

    # Decode outputs safely
    stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()

    if result.returncode != 0:
        # Show helpful error info but keep it concise
        if stderr:
            return f"❌ Ollama error: {stderr[:1200]}"
        return "❌ Ollama returned an error (no stderr output)."

    return stdout if stdout else "⚠️ Ollama returned an empty response."
