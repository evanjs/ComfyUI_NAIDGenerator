import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "OpusUsageTrackerNAID";

function outputText(output) {
    const value = output?.text;
    if (Array.isArray(value)) return value.map(String).join("\n");
    if (value != null) return String(value);
    return "Usage check completed.";
}

function compactLabel(text) {
    const percent = text.match(/^(\d+% remaining)/);
    if (percent) return percent[1];
    return text.toLowerCase().includes("error") || text.toLowerCase().includes("unavailable")
        ? "Usage unavailable"
        : "Check Usage";
}

function getPartialTargets(node) {
    const targets = new Set([String(node.id)]);
    const visited = new Set();
    const pending = [node];
    const links = app.graph?.links ?? {};

    while (pending.length) {
        const current = pending.shift();
        const currentId = String(current.id);
        if (visited.has(currentId)) continue;
        visited.add(currentId);

        for (const output of current.outputs ?? []) {
            for (const linkId of output.links ?? []) {
                const link = links[linkId];
                if (!link) continue;
                const downstream = app.graph.getNodeById(link.target_id);
                if (!downstream) continue;

                if (downstream.constructor?.nodeData?.output_node === true) {
                    targets.add(String(downstream.id));
                }
                pending.push(downstream);
            }
        }
    }

    return [...targets];
}

app.registerExtension({
    name: "Comfy.NAIDGenerator.OpusUsageTracker",

    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;

        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Check Usage";
        button.title = "Click to refresh your Opus V5 usage. Hover after checking for details.";
        button.style.cssText = "width:100%;padding:4px;cursor:pointer;";

        node.addDOMWidget("opus_usage_status", "div", button, {
            getMinHeight: () => 34,
            getHeight: () => 34,
            hideOnZoom: false,
        });

        const onExecuted = (event) => {
            const detail = event.detail ?? {};
            if (String(detail.node) !== String(node.id)) return;
            const info = outputText(detail.output);
            button.textContent = compactLabel(info);
            button.title = info;
            button.disabled = false;
        };

        const onExecutionError = (event) => {
            const detail = event.detail ?? {};
            if (String(detail.node_id) !== String(node.id)) return;
            const error = detail.exception_message || "Usage check failed; see ComfyUI errors.";
            button.textContent = "Usage unavailable";
            button.title = error;
            button.disabled = false;
        };

        api.addEventListener("executed", onExecuted);
        api.addEventListener("execution_error", onExecutionError);

        const previousOnRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
            api.removeEventListener("executed", onExecuted);
            api.removeEventListener("execution_error", onExecutionError);
            return previousOnRemoved?.apply(this, args);
        };

        button.addEventListener("click", async () => {
            if (button.disabled) return;
            button.disabled = true;
            button.textContent = "Checking…";
            button.title = "Fetching current usage from NovelAI…";

            try {
                const queued = await app.queuePrompt(0, 1, getPartialTargets(node));
                if (queued === false) {
                    button.textContent = "Check Usage";
                    button.title = "Could not queue the usage check. See ComfyUI errors.";
                    button.disabled = false;
                }
            } catch (error) {
                button.textContent = "Check Usage";
                button.title = `Could not queue usage check: ${error?.message ?? error}`;
                button.disabled = false;
            }
        });
    },
});
