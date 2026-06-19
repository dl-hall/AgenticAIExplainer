(function () {
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const modelSelect = $("#model-select");
    const btnLoadModel = $("#btn-load-model");
    const modelStatus = $("#model-status");
    const presetSelect = $("#preset-select");
    const promptInput = $("#prompt-input");
    const btnSend = $("#btn-send");
    const btnReset = $("#btn-reset");
    const contextContent = $("#context-content");
    const activityLog = $("#activity-log");
    const systemPromptEl = $("#system-prompt");
    const btnApplyPrompt = $("#btn-apply-prompt");
    const reasoningToggle = $("#reasoning-addendum-toggle");
    const temperatureEl = $("#temperature");
    const temperatureValueEl = $("#temperature-value");
    const tokenCount = $("#token-count");
    const loopCounter = $("#loop-counter");
    const statusIcon = $("#status-icon");
    const statusText = $("#status-text");

    let ws = null;
    let modelLoaded = false;
    let streamingEl = null;

    function connect() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        ws.onopen = () => setStatus("active", "Connected to server.");
        ws.onclose = () => {
            setStatus("error", "Disconnected. Refresh to reconnect.");
            modelLoaded = false;
            updateButtons();
        };
        ws.onerror = () => setStatus("error", "WebSocket error.");
        ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
    }

    function send(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(obj));
        }
    }

    function setStatus(level, text) {
        statusIcon.className = level;
        statusText.textContent = text;
    }

    function updateButtons() {
        btnSend.disabled = !modelLoaded;
    }

    function setLoopStep(stepId) {
        $$(".loop-step").forEach((el) => {
            el.classList.remove("active", "done");
        });
        if (!stepId) return;

        const stepOrder = ["step-observe", "step-think", "step-act"];
        const idx = stepOrder.indexOf(stepId);
        for (let i = 0; i < idx; i++) {
            $(`#${stepOrder[i]}`).classList.add("done");
        }
        $(`#${stepId}`).classList.add("active");
    }

    // --- Context panel ---

    function renderContext(messages, tools, promptText, reasoningAddendum, count) {
        contextContent.innerHTML = "";

        if (promptText) {
            addContextBlock("Full Prompt (raw tokens)", promptText, "system", true);
        }

        let toolsRendered = false;

        for (const msg of messages) {
            const role = msg.role;
            const contentText = extractText(msg.content);

            if (role === "system") {
                addContextBlock("System Prompt", contentText, "system");
                if (reasoningAddendum) {
                    addContextBlock(
                        "Reasoning Instructions (auto-added)",
                        reasoningAddendum,
                        "thinking"
                    );
                }
                if (!toolsRendered && tools && tools.length > 0) {
                    const toolText = JSON.stringify(tools, null, 2);
                    addContextBlock(
                        `Tool Definitions (${tools.length} tools)`,
                        toolText,
                        "tools"
                    );
                    toolsRendered = true;
                }
            } else if (role === "user") {
                addContextBlock("User Message", contentText, "user");
            } else if (role === "assistant") {
                if (msg.tool_calls && msg.tool_calls.length > 0) {
                    for (const tc of msg.tool_calls) {
                        const fn = tc.function || tc;
                        const args =
                            typeof fn.arguments === "string"
                                ? fn.arguments
                                : JSON.stringify(fn.arguments, null, 2);
                        addContextBlock(
                            `Tool Call: ${fn.name}`,
                            `Function: ${fn.name}\nArguments: ${args}`,
                            "tool-call"
                        );
                    }
                } else {
                    addContextBlock("Assistant Response", contentText, "assistant");
                }
            } else if (role === "tool") {
                addContextBlock(
                    `Tool Result: ${msg.name || "unknown"}`,
                    contentText,
                    "tool-result"
                );
            }
        }

        const realCount = typeof count === "number" ? count : 0;
        tokenCount.textContent = `${realCount.toLocaleString()} tokens`;

        contextContent.scrollTop = contextContent.scrollHeight;
    }

    function addContextBlock(title, content, cssClass, collapsed) {
        const block = document.createElement("div");
        block.className = `context-block ${cssClass}`;

        const header = document.createElement("div");
        header.className = "context-block-header";
        header.innerHTML = `<span>${title}</span><span class="toggle">${collapsed ? "&#9654;" : "&#9660;"}</span>`;

        const body = document.createElement("div");
        body.className = `context-block-body${collapsed ? " collapsed" : ""}`;
        body.textContent = content;

        header.addEventListener("click", () => {
            body.classList.toggle("collapsed");
            header.querySelector(".toggle").innerHTML = body.classList.contains(
                "collapsed"
            )
                ? "&#9654;"
                : "&#9660;";
        });

        block.appendChild(header);
        block.appendChild(body);
        contextContent.appendChild(block);
    }

    function extractText(content) {
        if (typeof content === "string") return content;
        if (Array.isArray(content)) {
            return content
                .filter((c) => c.type === "text")
                .map((c) => c.text)
                .join("\n");
        }
        return String(content);
    }

    // --- Activity log ---

    function addActivity(html, cssClass) {
        const el = document.createElement("div");
        el.className = `activity-entry ${cssClass || ""}`;
        el.innerHTML = html;
        activityLog.appendChild(el);
        activityLog.scrollTop = activityLog.scrollHeight;
        return el;
    }

    function startStreaming() {
        const wrapper = document.createElement("div");
        wrapper.className = "activity-entry streaming";
        wrapper.innerHTML =
            '<div class="label" style="color:var(--accent-purple)">LLM Output (token by token)</div>';
        streamingEl = document.createElement("span");
        streamingEl.id = "streaming-output";
        wrapper.appendChild(streamingEl);
        const cursor = document.createElement("span");
        cursor.className = "cursor-blink";
        wrapper.appendChild(cursor);
        activityLog.appendChild(wrapper);
        activityLog.scrollTop = activityLog.scrollHeight;
    }

    function appendToken(text) {
        if (!streamingEl) return;
        streamingEl.textContent += text;
        activityLog.scrollTop = activityLog.scrollHeight;
    }

    function stopStreaming() {
        const cursor = activityLog.querySelector(".cursor-blink");
        if (cursor) cursor.remove();
        streamingEl = null;
    }

    // --- Event handler ---

    function handleEvent(evt) {
        switch (evt.type) {
            case "model_loading":
                modelStatus.textContent = evt.text;
                modelStatus.className = "status-badge loading";
                setStatus("warning", evt.text);
                break;

            case "model_ready":
                modelLoaded = true;
                modelStatus.textContent = `${evt.model} ready`;
                modelStatus.className = "status-badge ready";
                setStatus("success", `Model '${evt.model}' loaded and ready.`);
                if (typeof evt.temperature === "number") {
                    temperatureEl.value = evt.temperature;
                    temperatureValueEl.textContent = evt.temperature.toFixed(2);
                }
                reasoningToggle.checked = !!evt.reasoning_addendum_enabled;
                updateButtons();
                break;

            case "loop_iteration":
                loopCounter.style.display = "";
                loopCounter.textContent = `Iteration ${evt.iteration} / ${evt.max}`;
                setLoopStep("step-observe");
                addActivity(
                    `<div class="label" style="color:var(--accent-blue)">Agent Loop — Iteration ${evt.iteration}</div>
                     <div>The harness begins a new cycle of the agent loop.</div>`,
                    "phase-label"
                );
                setStatus("active", `Agent loop — iteration ${evt.iteration}`);
                break;

            case "context_building":
                renderContext(evt.messages, evt.tools, evt.prompt_text, evt.reasoning_addendum, evt.token_count);
                setLoopStep("step-observe");
                addActivity(
                    `<div class="label" style="color:var(--role-system)">✨ Context Built</div>
                     <div>The harness has assembled the full prompt: system instructions, tool definitions, and conversation history. This entire block is sent to the LLM as input.</div>`,
                    "phase-label"
                );
                setStatus(
                    "active",
                    "Context assembled — sending to LLM..."
                );
                break;

            case "llm_start":
                setLoopStep("step-think");
                startStreaming();
                setStatus(
                    "active",
                    "The LLM is generating tokens one by one..."
                );
                break;

            case "token":
                appendToken(evt.text);
                break;

            case "llm_done":
                stopStreaming();
                setLoopStep("step-act");
                break;

            case "thinking_block":
                addActivity(
                    `<div class="label" style="color:var(--role-thinking)">🧠 Thinking Block</div>
                     <div>The reasoning model generated internal thinking before responding. This text is not shown to the user in a real app — it's the model "talking to itself" to break down the problem.</div>
                     <pre style="margin-top:6px;color:var(--role-thinking);opacity:0.8;font-size:12px">${escapeHtml(evt.text)}</pre>`,
                    "thinking"
                );
                break;

            case "tool_call_detected":
                for (const tc of evt.calls) {
                    addActivity(
                        `<div class="label" style="color:var(--role-tool-call)">🔧 Tool Call Detected</div>
                         <div>The LLM generated text that the harness recognized as a tool call. The LLM did not execute anything — it just "spoke" a structured request. The harness will now execute it.</div>
                         <div style="margin-top:4px"><strong>${escapeHtml(tc.name)}</strong>(${escapeHtml(JSON.stringify(tc.arguments))})</div>`,
                        "tool-call"
                    );
                }
                setStatus(
                    "active",
                    "Tool call detected — harness is executing..."
                );
                break;

            case "tool_executing":
                setStatus(
                    "active",
                    `Executing tool: ${evt.name}(${JSON.stringify(evt.arguments)})`
                );
                break;

            case "tool_result":
                addActivity(
                    `<div class="label" style="color:var(--role-tool-result)">✅ Tool Result</div>
                     <div>The harness executed <strong>${escapeHtml(evt.name)}</strong> and got a result. This result is now injected back into the context as a new message, and the agent loop continues.</div>
                     <pre style="margin-top:6px;font-size:12px">${escapeHtml(evt.result)}</pre>`,
                    "tool-result"
                );
                setStatus(
                    "active",
                    "Tool result received — adding to context for next iteration..."
                );
                break;

            case "response_complete":
                addActivity(
                    `<div class="label" style="color:var(--role-assistant)">💬 Final Response</div>
                     <div>The LLM generated a text response (not a tool call), so the agent loop ends. This is what the user would see.</div>
                     <pre style="margin-top:6px;font-size:13px;color:var(--text)">${escapeHtml(evt.text)}</pre>`,
                    "response"
                );
                setLoopStep(null);
                setStatus("success", "Agent finished — response delivered.");
                break;

            case "max_iterations":
                addActivity(
                    `<div class="label" style="color:var(--accent)">Max Iterations Reached</div>
                     <div>The agent hit the safety limit of ${evt.iterations} iterations.</div>`,
                    "error"
                );
                setLoopStep(null);
                setStatus("warning", "Agent stopped — max iterations.");
                break;

            case "reset":
                contextContent.innerHTML =
                    '<div class="empty-state">Send a prompt to see the context window build up.</div>';
                activityLog.innerHTML =
                    '<div class="empty-state">The agent loop will appear here as events unfold.</div>';
                loopCounter.style.display = "none";
                tokenCount.textContent = "0 tokens";
                setLoopStep(null);
                setStatus("active", "Conversation reset.");
                break;

            case "error":
                addActivity(
                    `<div class="label">Error</div><div>${escapeHtml(evt.text)}</div>`,
                    "error"
                );
                setStatus("error", evt.text);
                break;

            default:
                console.log("Unknown event:", evt);
        }
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // --- Controls ---

    btnLoadModel.addEventListener("click", () => {
        send({ action: "load_model", model: modelSelect.value });
        modelStatus.textContent = "Loading...";
        modelStatus.className = "status-badge loading";
        setStatus("warning", "Loading model...");
    });

    presetSelect.addEventListener("change", () => {
        if (presetSelect.value) {
            promptInput.value = presetSelect.value;
        }
    });

    btnSend.addEventListener("click", sendPrompt);
    promptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !btnSend.disabled) sendPrompt();
    });

    function sendPrompt() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        activityLog.querySelector(".empty-state")?.remove();

        send({ action: "send_prompt", prompt });
        promptInput.value = "";
        presetSelect.value = "";
    }

    btnReset.addEventListener("click", () => {
        send({ action: "reset" });
    });

    // --- Generation settings ---

    btnApplyPrompt.addEventListener("click", () => {
        send({ action: "set_system_prompt", system_prompt: systemPromptEl.value });
    });

    temperatureEl.addEventListener("input", () => {
        temperatureValueEl.textContent = parseFloat(temperatureEl.value).toFixed(2);
    });
    temperatureEl.addEventListener("change", () => {
        send({ action: "set_temperature", temperature: parseFloat(temperatureEl.value) });
    });

    reasoningToggle.addEventListener("change", () => {
        send({ action: "set_reasoning_addendum", enabled: reasoningToggle.checked });
    });

    // --- Init ---
    connect();
})();
