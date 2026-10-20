"""
FastAPI + SSE 流式 Chat 接口 Demo
=================================
能力：
  1. 流式输出（SSE）—— 打字机效果
  2. 重试         —— LLMClient 内置（超时/断网/429/5xx 自动指数退避重试）
  3. 限流         —— 用 slowapi 对每个客户端 IP 做速率限制（默认 10 次/分钟）

启动：
  uvicorn main:app --reload --port 8000

测试：
  curl -N -X POST http://localhost:8000/chat \
       -H "Content-Type: application/json" \
       -d '{"messages":[{"role":"user","content":"用一句话介绍你自己"}]}'
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 限流：slowapi（基于 limits 库），按客户端 IP 计数
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from llm_client import LLMClient

# ---------- 读取配置 ----------
load_dotenv()  # 读取项目根目录 .env

llm = LLMClient(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    model=os.environ["LLM_MODEL"],
    timeout=float(os.environ.get("LLM_TIMEOUT", "30")),
    max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
)

# 限流规则：每分钟最多 N 次
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")

# ---------- FastAPI 初始化 ----------
app = FastAPI(title="Chat SSE Demo")

# 限流器：key_func 决定"按谁计数"，这里按客户端 IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
# 超限时返回 429，并给出统一错误格式
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------- 静态文件（前端页面）----------
# FastAPI 默认不托管静态文件，必须手动挂载。
# 用 Path(__file__).parent 定位 static 目录，不依赖"当前运行目录"，更稳妥。
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


# ---------- 根路径 ----------
@app.get("/")
async def root():
    """根路径直接跳到前端演示页。"""
    return RedirectResponse("/static/index.html")


# ---------- 请求体模型 ----------
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]      # 对话历史，如 [{"role":"user","content":"你好"}]
    temperature: float = 0.7            # 采样温度
    max_tokens: Optional[int] = None    # 最大输出长度


# ---------- SSE 编码辅助 ----------
def sse_data(payload: dict) -> str:
    """把 dict 编码成一条 SSE 数据帧：data: <json>\n\n"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------- 核心接口 ----------
@app.post("/chat")
@limiter.limit(RATE_LIMIT)  # 限流装饰器，需要 request 参数在签名里
async def chat(request: Request, body: ChatRequest):
    """流式 Chat 接口，返回 text/event-stream。

    事件格式：
      data: {"delta": "..."}            # 正常输出的一小段
      data: {"delta": "..."}            # ...重复...
      event: error\ndata: {"message"}   # 出错时
      data: [DONE]                      # 结束标记
    """

    def gen():
        """同步生成器：FastAPI 会自动把它放进线程池，不阻塞事件循环。
        因为底层 LLMClient 是同步阻塞调用，所以用同步生成器更合适。"""
        try:
            # 内部已自动处理超时/重试；这里的 for 只是逐块转发
            for piece in llm.chat_stream(body.messages, body.temperature, body.max_tokens):
                yield sse_data({"delta": piece})
            yield "data: [DONE]\n\n"   # 正常结束标记

        except Exception as e:         # 重试耗尽或不可重试错误
            # 把错误也通过 SSE 事件发给前端，避免前端拿到"半截就断了"的困惑
            yield f"event: error\n{sse_data({'message': str(e)})}"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",          # 禁止缓存
            "X-Accel-Buffering": "no",            # 关掉 nginx 缓冲（部署时有用）
        },
    )


# ---------- 一个调试用的健康检查 ----------
@app.get("/health")
async def health():
    return {"status": "ok", "model": llm.model, "rate_limit": RATE_LIMIT}
