"""
LLM Client Implementations for Social RL Experiments

Supports:
- OllamaClient: Local Ollama server
- OpenAIClient: OpenAI-compatible APIs (vLLM, RunPod, etc.)
- MockClient: Testing without API calls
"""

from typing import Optional
import os


class MockClient:
    """Mock client for testing without actual API calls."""

    def __init__(self, **kwargs):
        self.call_count = 0

    def send_message(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        self.call_count += 1
        # Return mock responses based on agent type in prompt
        if "CES_Disengaged_Renter" in system_prompt or "disengaged" in system_prompt.lower():
            return "I guess. Not really sure it matters much."
        elif "CES_Rural_Conservative" in system_prompt or "conservative" in system_prompt.lower():
            return "This is exactly why we need less government interference. Let people keep more of what they earn."
        elif "CES_Urban_Progressive" in system_prompt or "progressive" in system_prompt.lower():
            return "We need systemic change. Housing should be a right, not a commodity."
        elif "CES_Suburban_Swing" in system_prompt:
            return "I see both sides. We need practical solutions that work for families like mine."
        else:
            return f"[Mock response {self.call_count}] Engaging with the discussion topic."


class OllamaClient:
    """Client for local Ollama server."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def send_message(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAIClient:
    """Client for OpenAI-compatible APIs (vLLM, RunPod, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        timeout: float = 120.0
    ):
        from openai import OpenAI

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.model = model
        self.base_url = base_url

        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
            # Add headers for common proxy issues
            client_kwargs["default_headers"] = {
                "ngrok-skip-browser-warning": "true"
            }

        self.client = OpenAI(**client_kwargs)

    def send_message(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


# For backwards compatibility
LLMClient = OpenAIClient
