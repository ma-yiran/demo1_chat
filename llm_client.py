"""
LLM 客户端封装：统一支持 DeepSeek / Qwen，自带超时、错误处理、自动重试。

设计要点：
- 两家平台都兼容 OpenAI 协议，所以共用同一个 OpenAI SDK 客户端。
- 关掉 SDK 自带的静默重试（max_retries=0），自己写"能看懂、可区分错误类型"的重试。
- 重试只覆盖"建立请求连接"阶段（_create），因为流式一旦开始吐数据就不能整体重试了。
"""

import json
import time
import random
from typing import List, Dict, Iterator, Any

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError


class LLMClient:
    """统一 LLM 客户端：文本 / 流式 / JSON 结构化，共用一套超时与重试。"""

    def __init__(
        self,
        api_key: str,            # 密钥，从环境变量读取
        base_url: str,           # 平台地址：DeepSeek / Qwen 不同
        model: str,              # 模型名：deepseek-chat / qwen-plus 等
        timeout: float = 30.0,   # 单次请求超时（秒）
        max_retries: int = 3,    # 最多重试几次（不含第一次）
    ):
        self.model = model
        self.max_retries = max_retries
        # 真正的 OpenAI 客户端。两家平台都兼容 OpenAI 协议，共用这一个。
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,     # 超时交给 SDK 处理
            max_retries=0,       # 关掉 SDK 自带的静默重试，用我们自己可控的重试
        )

    # ---------- 核心：建立请求 + 重试（所有方法共用） ----------
    def _create(self, messages, temperature, max_tokens,
                response_format=None, stream=False, tools=None):
        """只负责「发请求 + 重试」，返回 OpenAI 的 response 对象。
        流式时返回的 stream 对象还没开始读取，所以这一步重试是安全的。
        tools：传给模型的工具定义列表（function calling 用）。"""
        attempt = 0
        while True:
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,                          # 流式开关
                    response_format=response_format,        # JSON 结构化开关
                )
                if tools:            # 只有给了工具才带上，避免空参数
                    kwargs["tools"] = tools
                return self.client.chat.completions.create(**kwargs)
            except APITimeoutError as e:
                attempt = self._maybe_retry(attempt, "请求超时", e)
            except APIConnectionError as e:
                attempt = self._maybe_retry(attempt, "网络连接失败", e)
            except RateLimitError as e:
                attempt = self._maybe_retry(attempt, "触发限流(429)", e, wait=5.0)
            except APIError as e:
                if e.status_code and 500 <= e.status_code < 600:
                    attempt = self._maybe_retry(attempt, f"服务器错误({e.status_code})", e)
                else:
                    self._raise_unrecoverable(e)
            except Exception as e:
                raise RuntimeError(f"调用失败（未知错误）：{e}") from e

    # ---------- 对外方法 ----------

    def chat(self, messages, temperature=0.7, max_tokens=None) -> str:
        """普通文本（非流式），返回字符串。"""
        resp = self._create(messages, temperature, max_tokens)
        return resp.choices[0].message.content

    def chat_stream(self, messages, temperature=0.7, max_tokens=None) -> Iterator[str]:
        """流式输出（SSE）。用 for 遍历，逐块拿到文本片段。"""
        resp = self._create(messages, temperature, max_tokens, stream=True)
        for chunk in resp:  # 这里开始消费流
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta   # 每次吐一小段

    def chat_json(self, messages, temperature=0.7, max_tokens=None) -> Dict[str, Any]:
        """结构化 JSON（非流式），直接返回 dict。"""
        resp = self._create(
            messages, temperature, max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        try:
            return json.loads(text)   # 文本 → Python dict
        except json.JSONDecodeError as e:
            raise RuntimeError(f"模型返回的不是合法 JSON：{text[:200]}") from e

    # ---------- 工具调用（function calling） ----------
    def run_tool_loop(self, messages, tools, execute_tool,
                      temperature=0.7, max_tokens=None, max_rounds=5):
        """执行完整的「工具调用循环」，返回 (final_messages, call_logs)。

        流程（模型可以连续调用多个工具，直到不需要工具为止）：
          1. 非流式请求一次，带 tools
          2. 若模型返回 tool_calls → 交给 execute_tool 执行，把结果以 role="tool" 回填
          3. 再请求一次…… 重复，直到模型不再请求工具（max_rounds 兜底防死循环）
          4. 返回处理好的完整消息列表，供上层用 chat_stream 流式输出最终回答

        参数：
          messages    初始对话（role/content）
          tools       OpenAI 工具定义列表
          execute_tool(name, args) -> result   应用层提供的工具执行回调
        """
        history = [dict(m) for m in messages]   # 拷贝，避免污染调用方的列表
        call_logs: List[dict] = []              # 记录"调用了哪些工具"，供界面展示

        for _ in range(max_rounds):            # max_rounds 兜底，防止模型反复要工具死循环
            resp = self._create(history, temperature, max_tokens, tools=tools)
            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                break                          # 模型不需要工具了，循环结束

            # ① 把模型的"工具调用请求"作为 assistant 消息记入历史
            history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })

            # ② 逐个执行工具，把结果回填
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                call_logs.append({"name": name, "args": args})
                result = execute_tool(name, args)
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return history, call_logs

    # ---------- 重试 / 错误提示 ----------
    def _maybe_retry(self, attempt, reason, error, wait=None) -> int:
        """判断还能不能重试：能则等待并返回新的 attempt；不能则抛异常。"""
        attempt += 1
        if attempt > self.max_retries:
            raise RuntimeError(f"重试 {self.max_retries} 次仍失败（{reason}）") from error
        # 指数退避 + 抖动：第1次约2s，第2次约4s，第3次约8s，乘 0.5~1.5 随机数避免同时重试
        backoff = (2 ** attempt) * (0.5 + random.random())
        sleep_time = wait if wait is not None else backoff
        print(f"[重试 {attempt}/{self.max_retries}] {reason}，{sleep_time:.1f}s 后重试...")
        time.sleep(sleep_time)
        return attempt

    def _raise_unrecoverable(self, error: APIError) -> None:
        """不可重试的错误：翻译成人能看懂的中文提示再抛出。"""
        status = error.status_code
        if status == 401:
            raise RuntimeError("API Key 无效或过期，请检查 api_key") from error
        elif status == 402:
            raise RuntimeError("账户余额不足，请去平台充值") from error
        elif status == 404:
            raise RuntimeError("模型不存在，请检查 model 名称是否正确") from error
        elif status == 400:
            raise RuntimeError(f"请求参数错误：{error}") from error
        else:
            raise RuntimeError(f"调用失败（HTTP {status}）：{error}") from error
