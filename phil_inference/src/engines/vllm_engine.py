import os
import httpx
from openai import AsyncOpenAI

class LLMEngine:
    def __init__(self, config, component="brain"):
        """
        Khởi tạo kết nối tới Backend (vLLM/TGI)
        """
        backend_type = config[component]['active_backend']
        
        # Lấy URL endpoint từ config dựa trên backend đang chọn
        if component == "brain":
            base_url = config['brain']['endpoints'][backend_type]
        else:
            base_url = config[component]['endpoint']

        print(f">>> 🔗 Connected {component.upper()} to {backend_type} at {base_url}")
        
        # vLLM và TGI đều hỗ trợ chuẩn OpenAI API
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        self.model_name = config[component]['model_name']

    async def chat(self, messages: list, temperature=0.7):
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to AI Engine: {str(e)}"

    async def see(self, prompt: str, image_url: str):
        """Hàm dành riêng cho Vision"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        return await self.chat(messages)