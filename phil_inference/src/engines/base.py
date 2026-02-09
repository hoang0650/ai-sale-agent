from abc import ABC, abstractmethod

class LLMEngine(ABC):
    @abstractmethod
    async def chat(self, messages: list, system_prompt: str = None) -> str:
        """Hàm chat chung cho mọi backend"""
        pass

class EngineFactory:
    @staticmethod
    def get_engine(config):
        backend = config['brain']['backend']
        if backend == "vllm":
            from .vllm_engine import VLLMEngine
            return VLLMEngine(config['brain'])
        elif backend == "tgi":
            from .tgi_engine import TGIEngine
            return TGIEngine(config['brain'])
        elif backend == "llamacpp":
            from .llamacpp_engine import LlamaCppEngine
            return LlamaCppEngine(config['brain'])
        else:
            raise ValueError(f"Unknown backend: {backend}")