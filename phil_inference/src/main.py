import yaml
from fastapi import FastAPI, UploadFile, File, Form
from src.engines.vllm_engine import LLMEngine

# Load Config
with open("config/model_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(title="Phil AI Gateway")

# Khởi tạo các Engine
brain_engine = LLMEngine(cfg, "brain")
vision_engine = LLMEngine(cfg, "vision")

@app.post("/chat")
async def chat_endpoint(message: str = Form(...)):
    """Chat với DeepSeek"""
    response = await brain_engine.chat([{"role": "user", "content": message}])
    return {"response": response}

@app.post("/vision")
async def vision_endpoint(message: str = Form(...), image_url: str = Form(...)):
    """Nhìn ảnh với InternVL2"""
    response = await vision_engine.see(message, image_url)
    return {"response": response}

# Các endpoint Audio/TTS sẽ gọi sang service Whisper/F5 tương tự...