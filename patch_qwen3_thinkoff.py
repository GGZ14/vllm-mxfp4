#!/usr/bin/env python3
"""Bake: make the qwen3 reasoning parser agree with the chat template about whether thinking is on.

Symptom (measured 2026-08-21 on radiance 0.5.8 + froggeric v22.3, Qwen3.8-27B):

    POST /v1/chat/completions  chat_template_kwargs={"reasoning_effort": "off"}
    prompt "What is 17*23? Answer with just the number."
      -> finish_reason "stop", content = None, reasoning = "391"

The model answers correctly, but the answer lands in `reasoning` and `content` is null, so any
OpenAI-style client reading `content` sees an empty response.

Cause. `Qwen3Parser.__init__` decides its start state from exactly one key:

    self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

and feeds it to `qwen3_config(thinking=...)`, whose `initial_state` is REASONING when true. But the
froggeric template disables thinking -- i.e. emits `<think>\n\n</think>\n\n` into the PROMPT -- on
three more conditions the parser never looks at (template lines 18-35):

    reasoning_effort in ('none', 'off')            <- the case that bit us
    auto_disable_thinking_with_tools and tools
    an inline <|think_off|> tag in a system/developer/user message

When the prompt pre-closes the block, the model's OUTPUT contains no `</think>` at all. The parser
starts in REASONING, never sees a terminator, and files the whole stream as reasoning.

Note this is a parser/prompt disagreement, not a template bug: the parser only ever sees the output,
so it cannot observe that the prompt already closed thinking. It reproduces on ANY template that
pre-closes -- including stock Qwen -- whenever thinking is turned off by a route other than
`enable_thinking`.

Fix. Mirror the template's own decision from the same kwargs, before `qwen3_config()` consumes it.
That also repairs the streaming path, which keys off the identical `initial_state`.

Why not sniff the output instead: "no `</think>` present" cannot distinguish "thinking was off" from
"thinking was on and the response was truncated mid-thought". The second case must stay reasoning --
guessing there would corrupt the common truncation path to fix a rare one. Reading the kwargs the
template read is exact.

Not covered: the inline `<|think_off|>` tag. It lives in the message list, which `__init__` does not
receive; `extract_reasoning` gets a `request` and could see it, but that would fix only the
non-streaming half and leave streaming inconsistent, which is worse than a documented gap. Callers
wanting fast mode should pass `enable_thinking: false` or `reasoning_effort: "off"`.

Inherited for free by NemotronV3Parser and SeedOssParser, which subclass Qwen3Parser without
overriding __init__.

Idempotent; anchor-count-guarded; ast.parse guard before writing. NOOP once applied."""
import sysconfig
from pathlib import Path

from _patchlib import apply

LIB = Path(sysconfig.get_paths()["purelib"])
F = LIB / "vllm/parser/qwen3.py"

ANCHOR = (
    '        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}\n'
    '        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)'
)

SENTINEL = "[radiance] agree with the template about thinking"

NEW = (
    '        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}\n'
    '        # [radiance] agree with the template about thinking, not just enable_thinking.\n'
    '        # The froggeric template also pre-closes </think> in the PROMPT for\n'
    '        # reasoning_effort in {none, off} and for auto_disable_thinking_with_tools when\n'
    '        # tools are present. This parser only sees the OUTPUT, so a pre-closed block\n'
    '        # leaves no </think> to find and every token is filed as reasoning -- content\n'
    '        # comes back null and the answer hides in `reasoning`. Decide from the same\n'
    '        # kwargs the template read; sniffing the output cannot tell "thinking was off"\n'
    '        # from "truncated mid-thought", and the latter must stay reasoning.\n'
    '        _radiance_effort = chat_kwargs.get("reasoning_effort")\n'
    '        _radiance_effort = (\n'
    '            str(_radiance_effort).strip().lower()\n'
    '            if _radiance_effort is not None\n'
    '            else "medium"\n'
    '        )\n'
    '        self.thinking_enabled = bool(chat_kwargs.get("enable_thinking", True))\n'
    '        if _radiance_effort in ("none", "off"):\n'
    '            self.thinking_enabled = False\n'
    '        if chat_kwargs.get("auto_disable_thinking_with_tools") and tools:\n'
    '            self.thinking_enabled = False'
)

apply(F, ANCHOR, NEW, SENTINEL, "qwen3 parser: thinking-off agreement")
