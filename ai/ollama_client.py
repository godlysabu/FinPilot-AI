"""
ollama_client.py
----------------
Thin client for talking to a locally running Ollama server.

Ollama must be installed and running on the user's machine (default:
http://localhost:11434) with a model such as Qwen 3 already pulled
(e.g. `ollama pull qwen3:4b`). No API key, subscription, or internet
connection is required once the model has been downloaded once.

This module never calls the network on its own — it only talks to the
local Ollama server the user controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger("finpilot.ollama_client")

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_TIMEOUT_SECONDS = 120


@dataclass
class OllamaResponse:
    success: bool
    text: str = ""
    error: str = ""


def check_ollama_available(host: str = DEFAULT_HOST) -> OllamaResponse:
    """Ping the local Ollama server to check it's running and reachable."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        return OllamaResponse(success=True, text="Ollama is running.")
    except requests.exceptions.ConnectionError:
        return OllamaResponse(
            success=False,
            error=(
                "Could not connect to Ollama. Make sure Ollama is installed and running "
                f"on this machine (default address: {host}). Start it with the Ollama app "
                "or by running `ollama serve` in a terminal."
            ),
        )
    except requests.exceptions.Timeout:
        return OllamaResponse(success=False, error="Connection to Ollama timed out.")
    except requests.exceptions.RequestException as exc:  # noqa: BLE001
        return OllamaResponse(success=False, error=f"Unexpected error contacting Ollama: {exc}")


def list_available_models(host: str = DEFAULT_HOST) -> list[str]:
    """Return the names of models currently pulled/available in the local Ollama installation."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except requests.exceptions.RequestException:
        return []


def generate_insight(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> OllamaResponse:
    """
    Send a prompt to the local Ollama server and return the generated text.
    Uses the non-streaming /api/generate endpoint for simplicity.
    """
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        if not text:
            return OllamaResponse(success=False, error="Ollama returned an empty response.")
        return OllamaResponse(success=True, text=text)

    except requests.exceptions.ConnectionError:
        return OllamaResponse(
            success=False,
            error=(
                "Could not connect to Ollama. Make sure Ollama is installed and running "
                f"on this machine (default address: {host})."
            ),
        )
    except requests.exceptions.Timeout:
        return OllamaResponse(
            success=False,
            error=f"Ollama took longer than {timeout}s to respond. Try a smaller/faster model, or increase the timeout.",
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 404:
            return OllamaResponse(
                success=False,
                error=(
                    f"Model '{model}' was not found in Ollama. Pull it first with: "
                    f"`ollama pull {model}`"
                ),
            )
        return OllamaResponse(success=False, error=f"Ollama returned an HTTP error ({status}).")
    except requests.exceptions.RequestException as exc:  # noqa: BLE001
        logger.exception("Unexpected error calling Ollama")
        return OllamaResponse(success=False, error=f"Unexpected error contacting Ollama: {exc}")
    except ValueError:  # JSON decoding failure
        return OllamaResponse(success=False, error="Ollama returned a response that couldn't be parsed as JSON.")
