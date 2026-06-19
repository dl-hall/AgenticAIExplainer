import json
import re
import asyncio
import string
from dataclasses import dataclass, field

from app.model_manager import ModelManager, MAX_CONTEXT, NATIVE_CONTEXT
from app.tools import TOOL_DEFINITIONS, execute_tool

SYSTEM_PROMPT = """You are a helpful assistant. You have access to tools that let you perform calculations, read files, and list available files. Use these tools when needed to answer the user's questions accurately.

When you need information that might be in a file, first list the available files, then read the relevant one."""

# Mistral's standard reasoning prompt. Appended to the system prompt (as a
# distinct segment) when the reasoning addendum toggle is on. Its [THINK]/[/THINK]
# markers align with `_split_thinking` so the thinking renders as its own block.
REASONING_ADDENDUM = """# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response."""

# Per-model default temperatures (Mistral recommendation). Reset to these on
# model load/switch; user may override afterwards.
DEFAULT_TEMPERATURES = {"instruct": 0.1, "reasoning": 0.7}

MAX_ITERATIONS = 5

_call_counter = 0


def _make_call_id():
    global _call_counter
    _call_counter += 1
    chars = string.ascii_letters + string.digits
    result = []
    n = _call_counter
    for _ in range(9):
        result.append(chars[n % len(chars)])
        n //= len(chars)
    return "".join(result)


@dataclass
class AgentEvent:
    type: str
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {"type": self.type, **self.data}


class Agent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.messages = []
        self.tools_enabled = True
        self.list_files_enabled = True
        self.system_prompt = SYSTEM_PROMPT
        self.temperature = DEFAULT_TEMPERATURES["instruct"]
        self.reasoning_addendum_enabled = False

    def reset(self):
        # Only the conversation is cleared. Generation settings (system prompt,
        # temperature, reasoning toggle) deliberately persist across resets.
        self.messages = []

    def apply_model_defaults(self, model_key: str):
        """Reset generation settings to the loaded model's defaults.
        Returns (temperature, reasoning_addendum_enabled) so the caller can
        emit them to the client."""
        self.temperature = DEFAULT_TEMPERATURES.get(model_key, 0.7)
        self.reasoning_addendum_enabled = (model_key == "reasoning")
        return self.temperature, self.reasoning_addendum_enabled

    def get_active_tools(self):
        if not self.tools_enabled:
            return []
        tools = [t for t in TOOL_DEFINITIONS]
        if not self.list_files_enabled:
            tools = [t for t in tools if t["function"]["name"] != "list_files"]
        return tools

    async def run(self, user_prompt: str, emit):
        """Run the agent loop. `emit` is an async callable that sends AgentEvent to the client."""

        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        })

        tools = self.get_active_tools()

        for iteration in range(1, MAX_ITERATIONS + 1):
            await emit(AgentEvent("loop_iteration", {"iteration": iteration, "max": MAX_ITERATIONS}))

            addendum = REASONING_ADDENDUM if self.reasoning_addendum_enabled else None
            model_system_text = self.system_prompt + (f"\n\n{addendum}" if addendum else "")

            # full_messages -> model input (system prompt + addendum combined).
            full_messages = [
                {"role": "system", "content": [{"type": "text", "text": model_system_text}]},
                *self.messages,
            ]
            # display_messages -> context window (user's prompt only; the addendum
            # is shown as its own block so the editable prompt is never duplicated).
            display_messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                *self.messages,
            ]

            # Tokenize once: the same input_ids feed both display (decoded back to
            # text, counted for the badge) and generation below.
            tokenized = self.model_manager.tokenize(full_messages, tools=tools or None)
            input_ids = tokenized["input_ids"]
            token_count = input_ids.shape[1]
            prompt_text = self.model_manager.decode_prompt(input_ids)

            await emit(AgentEvent("context_building", {
                "messages": _serialize_messages(display_messages),
                "tools": tools,
                "prompt_text": prompt_text,
                "token_count": token_count,
                "max_context": MAX_CONTEXT,        # carried for bucket 06; not displayed yet
                "native_context": NATIVE_CONTEXT,
                "reasoning_addendum": addendum,
            }))

            await emit(AgentEvent("llm_start", {}))

            generated_text = ""
            loop = asyncio.get_event_loop()

            async def stream_tokens():
                nonlocal generated_text
                queue = asyncio.Queue()

                def on_token(token_text):
                    loop.call_soon_threadsafe(queue.put_nowait, token_text)

                def run_generation():
                    nonlocal generated_text
                    generated_text = self.model_manager.generate_streaming(
                        tokenized, on_token=on_token, temperature=self.temperature
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, None)

                import threading
                thread = threading.Thread(target=run_generation, daemon=True)
                thread.start()

                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    is_thinking = "[THINK]" in generated_text or generated_text.strip().startswith("<think")
                    await emit(AgentEvent("token", {"text": token, "is_thinking": is_thinking}))

                thread.join()

            await stream_tokens()

            await emit(AgentEvent("llm_done", {"full_text": generated_text}))

            clean_text = generated_text.replace("</s>", "").strip()
            thinking_text, response_text = _split_thinking(clean_text)

            if thinking_text:
                await emit(AgentEvent("thinking_block", {"text": thinking_text}))

            tool_calls = _parse_tool_calls(generated_text)

            if tool_calls:
                call_ids = [_make_call_id() for _ in tool_calls]

                await emit(AgentEvent("tool_call_detected", {
                    "calls": [{"name": tc["name"], "arguments": tc["arguments"]} for tc in tool_calls],
                }))

                self.messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_ids[i],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                })

                for i, tc in enumerate(tool_calls):
                    await emit(AgentEvent("tool_executing", {"name": tc["name"], "arguments": tc["arguments"]}))

                    result = execute_tool(tc["name"], tc["arguments"])

                    await emit(AgentEvent("tool_result", {"name": tc["name"], "result": result}))

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call_ids[i],
                        "name": tc["name"],
                        "content": result,
                    })

                continue

            clean_response = response_text.replace("</s>", "").strip()
            self.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": clean_response}],
            })
            await emit(AgentEvent("response_complete", {"text": clean_response}))
            return

        await emit(AgentEvent("max_iterations", {"iterations": MAX_ITERATIONS}))


def _serialize_messages(messages):
    """Make messages JSON-safe for sending to the frontend."""
    serialized = []
    for msg in messages:
        entry = {"role": msg["role"]}
        content = msg.get("content", "")
        if isinstance(content, list):
            entry["content"] = content
        elif isinstance(content, str):
            entry["content"] = [{"type": "text", "text": content}]
        else:
            entry["content"] = [{"type": "text", "text": str(content)}]

        if "tool_calls" in msg:
            entry["tool_calls"] = msg["tool_calls"]
        if "name" in msg:
            entry["name"] = msg["name"]
        if "tool_call_id" in msg:
            entry["tool_call_id"] = msg["tool_call_id"]

        serialized.append(entry)
    return serialized


def _split_thinking(text):
    """Split thinking blocks from the response text.
    Handles <think>...</think> tags and [THINK]...[/THINK] markers."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        response = text[:think_match.start()] + text[think_match.end():]
        return thinking, response.strip()

    think_match = re.search(r"\[THINK\](.*?)\[/THINK\]", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        response = text[:think_match.start()] + text[think_match.end():]
        return thinking, response.strip()

    return None, text


def _parse_tool_calls(text):
    """Parse tool calls from the model's output.
    Mistral models use [TOOL_CALLS]name[ARGS]{json} format."""
    tool_calls = []

    # Format: [TOOL_CALLS]function_name[ARGS]{"key": "value"}
    for match in re.finditer(r"\[TOOL_CALLS\]\s*(\w+)\[ARGS\]\s*(\{.*?\})(?:</s>|$)", text, re.DOTALL):
        try:
            name = match.group(1)
            args = json.loads(match.group(2))
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            pass
    if tool_calls:
        return tool_calls

    # Fallback: [TOOL_CALLS] followed by JSON array
    tc_match = re.search(r"\[TOOL_CALLS\]\s*(\[.*\])", text, re.DOTALL)
    if tc_match:
        try:
            calls = json.loads(tc_match.group(1))
            for call in calls:
                tool_calls.append({
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments", {}),
                })
        except json.JSONDecodeError:
            pass
        return tool_calls

    # Fallback: bare JSON tool call
    json_match = re.search(r'\{"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{[^}]*\})', text, re.DOTALL)
    if json_match:
        try:
            name = json_match.group(1)
            args = json.loads(json_match.group(2))
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            pass

    return tool_calls
