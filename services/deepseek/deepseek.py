import os
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

class DeepSeekEngine:
    def __init__(self, config: Dict[str, Any]):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self.model = config.get("model", "deepseek-chat")
        self.max_tokens = config.get("max_tokens", 4096)
        self.thinking = config.get("thinking", {"type": "enabled"})
        self.reasoning_effort = config.get("reasoning_effort", "medium")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            max_tokens=self.max_tokens,
            # reasoning_effort=self.reasoning_effort,
            extra_body={"thinking": self.thinking} if self.thinking else None
        )
        return response.choices[0].message.content
