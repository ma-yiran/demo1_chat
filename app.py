"""
Gradio Web 应用：流式聊天 + 工具调用，支持 DeepSeek / 千问(Qwen) 双模型切换
==========================================================================
功能：
  1. 流式聊天（打字机效果）
  2. 工具调用（function calling）：当前时间 / 计算器 / 模拟天气
  3. 双模型切换：顶部下拉框选择 DeepSeek 或 Qwen，各自独立 base_url/api_key/model

启动：
  python app.py        （或 gradio app.py）
  浏览器打开 http://127.0.0.1:7860

配置（.env）：
  DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
  DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL / DASHSCOPE_MODEL
"""

import ast
import datetime
import json
import os
from zoneinfo import ZoneInfo

import gradio as gr
from dotenv import load_dotenv

from llm_client import LLMClient


# =====================================================================
# 一、模型配置（DeepSeek / Qwen 双模型）
# =====================================================================
load_dotenv()  # 读取项目根目录 .env

MODELS = {
    "DeepSeek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "Qwen 千问": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "base_url": os.environ.get("DASHSCOPE_BASE_URL",
                                   "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": os.environ.get("DASHSCOPE_MODEL", "qwen-plus"),
    },
}

# 给每个模型缓存一个客户端，避免每次请求都新建连接
_client_cache = {}


def get_client(model_key: str) -> LLMClient:
    """按模型名取（缓存）客户端。"""
    if model_key not in _client_cache:
        cfg = MODELS[model_key]
        if not cfg["api_key"]:
            raise RuntimeError(f"未配置 {model_key} 的 API Key，请在 .env 里填写")
        _client_cache[model_key] = LLMClient(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
        )
    return _client_cache[model_key]


# =====================================================================
# 二、工具定义（演示用 3 个工具）
# =====================================================================
def tool_get_current_time(tz: str = "Asia/Shanghai") -> dict:
    """获取指定时区的当前时间。"""
    return {"time": datetime.datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M:%S %Z")}


def tool_calculator(expression: str) -> dict:
    """安全地计算一个数学表达式（只允许数字和四则运算）。"""
    # 用 ast 做白名单校验，避免 eval 执行任意代码（安全）
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
               ast.USub, ast.UAdd, ast.Pow)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("表达式只支持数字和四则运算")
    return {"result": eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})}


def tool_get_weather(city: str) -> dict:
    """查询城市天气（demo 用模拟数据，不连真实天气 API）。"""
    fake = {"杭州": "晴 22°C", "北京": "多云 18°C", "上海": "小雨 20°C", "深圳": "晴 26°C"}
    return {"city": city, "weather": fake.get(city, f"暂无 {city} 的数据")}


# 工具名 -> 可调用函数
TOOL_FUNCTIONS = {
    "get_current_time": tool_get_current_time,
    "calculator": tool_calculator,
    "get_weather": tool_get_weather,
}

# OpenAI 格式的工具定义（喂给模型的 schema）
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取指定时区的当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "tz": {"type": "string", "description": "时区，如 Asia/Shanghai，默认 Asia/Shanghai"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算一个数学表达式，例如 '12 * 7 + 3'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询一个城市的天气（模拟数据）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 杭州"}
                },
                "required": ["city"],
            },
        },
    },
]


def execute_tool(name: str, args: dict):
    """应用层工具执行回调：按名字调用对应函数，找不到就报错。"""
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"未知工具：{name}")
    return TOOL_FUNCTIONS[name](**args)


# =====================================================================
# 三、Gradio 聊天核心逻辑
# =====================================================================
def respond(user_input: str, history, model_key: str, use_tools: bool):
    """流式回复生成器。history 是 Gradio Chatbot(type="messages") 的列表。

    关键：真实模型"首 token"要 1~3 秒，回答短的句子几秒就出完。
    所以必须在调用模型前先 yield 一次"思考中"，否则界面会长时间静止，
    看起来像"非流式、一次性出结果"。
    """

    # 空输入直接返回
    if not user_input.strip():
        yield history, ""
        return

    # 初始化
    history = list(history or [])
    history.append({"role": "user", "content": user_input})

    # 转成发给模型的 messages（只保留纯对话，去掉展示用的中间提示）
    llm_messages = [
        {"role": h["role"], "content": h.get("content", "")}
        for h in history
        if h.get("role") in ("user", "assistant")
    ]

    try:
        llm = get_client(model_key)

        # 立即给反馈："思考中"占位 → 界面马上有反应，不再干等
        history.append({"role": "assistant", "content": "🤔 思考中..."})
        yield list(history), ""

        # ---- 工具调用阶段 ----
        call_logs = []
        if use_tools:
            llm_messages, call_logs = llm.run_tool_loop(llm_messages, TOOLS_SPEC, execute_tool)
            # 把"调用了哪些工具"以提示形式展示给用户（不进模型上下文）
            for log in call_logs:
                history.append({
                    "role": "assistant",
                    "content": f"🛠️ 已调用工具 `{log['name']}`，参数 {json.dumps(log['args'], ensure_ascii=False)}",
                })
                yield list(history), ""

        # ---- 流式输出最终回答（另起一条 assistant，逐段填充）----
        history.append({"role": "assistant", "content": ""})  # 占位，逐段填充
        assistant_text = ""
        for piece in llm.chat_stream(llm_messages):
            assistant_text += piece
            history[-1]["content"] = assistant_text
            yield list(history), ""

        # 如果模型一个字都没说，给个兜底提示
        if not assistant_text.strip():
            history[-1]["content"] = "[模型未返回内容]"
            yield list(history), ""

    except Exception as e:
        history.append({"role": "assistant", "content": f"[错误] {e}"})
        yield list(history), ""


# =====================================================================
# 四、界面搭建
# =====================================================================
# 界面搭建（Gradio 6.x：Chatbot 默认即为 messages 格式，无需 type 参数；theme 放到 launch）
with gr.Blocks(title="LLM Chat + Tool Calling") as demo:
    gr.Markdown("## 🤖 流式聊天 + 工具调用 Demo\nDeepSeek / 千问双模型切换，支持当前时间、计算器、天气模拟工具。")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODELS.keys()), value="DeepSeek", label="模型选择", scale=1,
        )
        tools_checkbox = gr.Checkbox(value=True, label="启用工具调用", scale=1)

    chatbot = gr.Chatbot(height=450, label="对话")

    with gr.Row():
        msg_box = gr.Textbox(placeholder="例如：现在几点了？  或  帮我算 12*7+3  （回车发送）", scale=4, show_label=False)
        send_btn = gr.Button("发送", scale=1)

    clear_btn = gr.Button("清空对话")

    # 回车 + 按钮都触发，输出：聊天记录 + 清空输入框
    send_btn.click(respond, [msg_box, chatbot, model_dropdown, tools_checkbox], [chatbot, msg_box])
    msg_box.submit(respond, [msg_box, chatbot, model_dropdown, tools_checkbox], [chatbot, msg_box])
    clear_btn.click(lambda: ([], ""), None, [chatbot, msg_box])

    gr.Markdown(
        "**试试这些话**："
        "「现在几点了」「杭州天气怎么样」「帮我算 (123*45-67)/8」——模型会调用对应工具再回答。"
    )

# 流式输出需要 queue
demo.queue()

if __name__ == "__main__":
    # 用 __main__ 保护：import 本模块（如测试、复用）时不会启动服务
    demo.launch(theme=gr.themes.Soft())
