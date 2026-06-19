import json
import re
import asyncio
import string
from dataclasses import dataclass, field

from app.model_manager import ModelManager, MAX_CONTEXT, NATIVE_CONTEXT
from app.tools import TOOL_DEFINITIONS, execute_tool, tool_names

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
        # Per-tool enabled set, keyed by the names in TOOL_DEFINITIONS. All on
        # by default; the presenter toggles individual tools from the UI.
        self.enabled_tools = set(tool_names())
        self.system_prompt = SYSTEM_PROMPT
        self.temperature = DEFAULT_TEMPERATURES["instruct"]
        self.reasoning_addendum_enabled = False
        self.max_iterations = MAX_ITERATIONS

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
        """Filter TOOL_DEFINITIONS down to the per-tool enabled set. An empty
        set returns [], which the tokenize path treats as 'no tools'."""
        return [t for t in TOOL_DEFINITIONS
                if t["function"]["name"] in self.enabled_tools]

    def _build_messages(self):
        """Build (model_input_messages, display_messages, addendum).
        model input combines system prompt + reasoning addendum; display keeps
        the editable system prompt separate so the addendum shows as its own
        block and is never duplicated."""
        addendum = REASONING_ADDENDUM if self.reasoning_addendum_enabled else None
        model_system_text = self.system_prompt + (f"\n\n{addendum}" if addendum else "")
        full_messages = [
            {"role": "system", "content": [{"type": "text", "text": model_system_text}]},
            *self.messages,
        ]
        display_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            *self.messages,
        ]
        return full_messages, display_messages, addendum

    async def _emit_context(self, tools, emit):
        """Tokenize the current conversation and emit the context_building event.
        Returns the tokenized input for generation. `tools` is the active tool
        list ([] disables tools for this generation)."""
        full_messages, display_messages, addendum = self._build_messages()

        # Tokenize once: the same input_ids feed both display (decoded back to
        # text, counted for the badge) and generation.
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
        return tokenized

    async def _generate_and_stream(self, tokenized, emit):
        """Run generation in a worker thread, emitting llm_start, per-token, and
        llm_done events. Returns the full generated text. Shared by the normal
        loop body and the forced final-answer pass."""
        await emit(AgentEvent("llm_start", {}))

        generated_text = ""
        loop = asyncio.get_event_loop()
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

        await emit(AgentEvent("llm_done", {"full_text": generated_text}))
        return generated_text

    async def run(self, user_prompt: str, emit):
        """Run the agent loop. `emit` is an async callable that sends AgentEvent to the client."""

        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        })

        tools = self.get_active_tools()

        for iteration in range(1, self.max_iterations + 1):
            await emit(AgentEvent("loop_iteration", {"iteration": iteration, "max": self.max_iterations}))

            tokenized = await self._emit_context(tools, emit)
            generated_text = await self._generate_and_stream(tokenized, emit)

            # Parse tool calls first, then strip the call markup out of the text
            # used for display, so a call emitted inside a thinking block is both
            # honored AND not shown as raw [TOOL_CALLS]... prose.
            tool_calls = _parse_tool_calls(generated_text)
            display_text = _strip_tool_call_markup(generated_text).replace("</s>", "").strip()
            thinking_text, response_text = _split_thinking(display_text)

            if thinking_text:
                await emit(AgentEvent("thinking_block", {"text": thinking_text}))

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

        # Safety cap reached. Instead of a dead-end, do one final generation with
        # tools disabled so the user still gets a real best-effort answer.
        await self._force_final_answer(emit)

    async def _force_final_answer(self, emit):
        """Run after the iteration cap is hit: generate one answer with tools
        disabled (the model must respond in prose) and emit it as the response,
        flagged so the UI can note the cap was reached."""
        await emit(AgentEvent("max_iterations", {
            "iterations": self.max_iterations,
            "forcing_answer": True,
        }))

        tokenized = await self._emit_context([], emit)  # [] -> tools disabled
        generated_text = await self._generate_and_stream(tokenized, emit)

        display_text = _strip_tool_call_markup(generated_text).replace("</s>", "").strip()
        thinking_text, response_text = _split_thinking(display_text)

        if thinking_text:
            await emit(AgentEvent("thinking_block", {"text": thinking_text}))

        clean_response = response_text.replace("</s>", "").strip()
        self.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": clean_response}],
        })
        await emit(AgentEvent("response_complete", {"text": clean_response, "capped": True}))


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
    Handles <think>/<thinking> tags and [THINK]...[/THINK] markers.

    The reasoning model is inconsistent about its delimiter: the prompt asks
    for [THINK]...[/THINK], but it sometimes emits <think>...</think> or
    <thinking>...</thinking> instead. Accept all three so thinking never leaks
    into the displayed response."""
    think_match = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", text, re.DOTALL)
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


# Locates each Mistral call header: [TOOL_CALLS]<name>[ARGS] . The JSON object
# that follows is decoded separately (balanced parse), NOT by regex — the model
# often keeps generating after a call (more thinking, [/THINK], whitespace), and
# tool arguments can contain nested braces, both of which defeat a regex anchor.
_TOOL_CALL_HEADER = re.compile(r"\[TOOL_CALLS\]\s*(\w+)\s*\[ARGS\]\s*")


def _find_tool_call_spans(text):
    """Find every [TOOL_CALLS]name[ARGS]{json} call in `text`.

    Returns a list of (start, end, name, arguments) tuples where start/end are
    character offsets spanning the full call markup (header + JSON, plus a
    trailing </s> if present), so callers can both read the call and strip the
    markup out of displayed text. The JSON object is extracted with a balanced
    decode (json.JSONDecoder().raw_decode), so it is independent of whatever
    trails the call and correctly handles nested-brace arguments."""
    decoder = json.JSONDecoder()
    spans = []
    for match in _TOOL_CALL_HEADER.finditer(text):
        name = match.group(1)
        json_start = match.end()
        try:
            args, consumed = decoder.raw_decode(text, json_start)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(args, dict):
            continue
        end = consumed
        # Swallow an immediately-following EOS token so it isn't left as prose.
        if text[end:end + 4] == "</s>":
            end += 4
        spans.append((match.start(), end, name, args))
    return spans


def _parse_tool_calls(text):
    """Parse tool calls from the model's output.
    Mistral models use [TOOL_CALLS]name[ARGS]{json} format. Returns a list of
    {"name", "arguments"} dicts (possibly empty)."""
    return [{"name": name, "arguments": args}
            for (_, _, name, args) in _find_tool_call_spans(text)]


def _strip_tool_call_markup(text):
    """Remove [TOOL_CALLS]...{json} call markup from `text` so it is not shown
    as prose in the thinking/response display. Operates on the spans found by
    _find_tool_call_spans, removing them back-to-front to keep offsets valid."""
    spans = _find_tool_call_spans(text)
    for start, end, _, _ in reversed(spans):
        text = text[:start] + text[end:]
    return text
