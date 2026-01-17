from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os, json
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ---------------- BASE MODEL ----------------
# Model đã train sẵn (hoặc base Qwen/Gemma)
MODEL_PATH = "./base-ai-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

# Embedding model
embedder = SentenceTransformer("intfloat/e5-small")

# ---------------- DATASET UPLOAD ----------------

class Product(BaseModel):
    name: str
    price: str
    description: str
    use_case: str = ""  # Optional

class DatasetPayload(BaseModel):
    tenant_id: str
    products: List[Product]

def product_to_text(p):
    return f"Tên: {p['name']}\nGiá: {p['price']}\nMô tả: {p['description']}\nPhù hợp: {p.get('use_case','')}"

@app.post("/dataset")
def upload_dataset(data: DatasetPayload):
    texts = [product_to_text(p.dict()) for p in data.products]
    vectors = embedder.encode(texts)

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    os.makedirs("vector", exist_ok=True)
    # Lưu index
    faiss.write_index(index, f"vector/{data.tenant_id}.index")
    # Lưu metadata
    with open(f"vector/{data.tenant_id}.json", "w") as f:
        json.dump([p.dict() for p in data.products], f)

    return {"status": "ok", "num_products": len(texts)}

# ---------------- CHAT API ----------------

class ChatPayload(BaseModel):
    tenant_id: str
    question: str

@app.post("/chat")
def chat(data: ChatPayload):
    index_path = f"vector/{data.tenant_id}.index"
    meta_path = f"vector/{data.tenant_id}.json"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return {"error": "Tenant data not found"}

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(meta_path) as f:
        products = json.load(f)

    # Encode question
    q_vec = embedder.encode([data.question])
    D, I = index.search(q_vec, k=3)  # top 3 relevant

    # Build context
    context = "\n---\n".join([product_to_text(products[i]) for i in I[0]])

    # Build prompt chốt mềm
    prompt = f"""
Bạn là AI tư vấn bán hàng theo phong cách CHỐT MỀM.
Quy tắc:
- Luôn xác nhận nhu cầu khách
- Không ép mua
- Kết thúc bằng câu hỏi gợi mở
- Chỉ sử dụng dữ liệu được cung cấp bên dưới

Thông tin sản phẩm:
{context}

Câu hỏi khách:
{data.question}
"""

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer.strip()}
