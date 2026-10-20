# LLM Chat Demo（FastAPI + Gradio）

一个**可直接运行**的示例项目，包含两套入口，共用同一个 `LLMClient`：

- **`main.py`** — FastAPI + SSE 流式接口，自带**流式输出 / 重试 / 限流**
- **`app.py`** — Gradio Web 应用：**流式聊天 + 工具调用**，支持 **DeepSeek / 千问双模型切换**

## 目录结构

```
fastapi-chat-demo/
├── app.py             # Gradio 应用：流式聊天 + 工具调用 + 双模型切换
├── main.py            # FastAPI 入口 + SSE 接口 + 限流
├── llm_client.py      # LLM 客户端（超时 + 错误处理 + 重试 + 工具调用）
├── static/index.html  # FastAPI 版的前端打字机演示页
├── requirements.txt
└── .env.example       # 环境变量模板
```

## 快速开始

```bash
# 1. 安装依赖（建议用虚拟环境）
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env        # 编辑 .env，填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY

# 3. 启动 Gradio 应用（推荐先看这个）
python app.py
# 浏览器打开 http://127.0.0.1:7860

# 或启动 FastAPI 版
uvicorn main:app --reload --port 8000
```

---

## 一、Gradio 应用（app.py）

界面顶部有两个控件：
- **模型选择**下拉框：DeepSeek / Qwen 千问，切换即换 `base_url / api_key / model`
- **启用工具调用**开关：打开后模型可调用工具

**内置 3 个演示工具**：

| 工具 | 说明 | 试试说 |
|---|---|---|
| `get_current_time` | 返回当前时间 | 「现在几点了」 |
| `calculator` | 安全计算四则运算 | 「帮我算 (123*45-67)/8」 |
| `get_weather` | 模拟天气（不连真实 API） | 「杭州天气怎么样」 |

**工具调用流程**（`llm_client.py` 的 `run_tool_loop`）：
```
用户提问 → 模型请求工具 → 应用执行工具 → 结果回填 → 模型基于结果流式回答
```

> 工具执行结果只在内部传给模型，界面上会显示一条「🛠️ 调用工具 xxx」的提示，方便观察。

## 二、FastAPI 版（main.py）

见下方「SSE / 重试 / 限流」说明。

### 1. SSE 流式
接口返回 `text/event-stream`：
```
data: {"delta": "你"}
data: {"delta": "好"}
...
data: [DONE]
```
出错发 `event: error` 帧。

### 2. 重试
`llm_client.py` 的 `_create()`：
- **可重试**：超时、断网、限流(429)、服务器 5xx → 指数退避（1s/2s/4s…）
- **不可重试**：401 / 402 / 404 / 400 → 抛中文报错
- 只重试"建立连接阶段"，流式开始吐数据后不整体重试

### 3. 限流
`slowapi` 按 IP 计数，`.env` 里 `RATE_LIMIT=10/minute`。

## 配置说明（.env）

| 变量 | 说明 | 示例 |
|---|---|---|
| `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` | 两平台密钥 | sk-... |
| `DEEPSEEK_BASE_URL` | DeepSeek 地址 | `https://api.deepseek.com` |
| `DASHSCOPE_BASE_URL` | 千问兼容地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DEEPSEEK_MODEL` / `DASHSCOPE_MODEL` | 模型名 | `deepseek-chat` / `qwen-plus` |
| `LLM_*` | FastAPI 版通用配置 | 超时 / 重试 / 限流 |

## 验证小贴士
- 测工具调用：不开工具开关时问「杭州天气怎么样」模型会瞎编；打开后会走 `get_weather` 工具。
- 测重试：把某个 `BASE_URL` 改成错误地址，模型侧会打印重试日志。
- 模型没配 key 时，Gradio 界面会显示 `[错误] 未配置 xx 的 API Key`。
