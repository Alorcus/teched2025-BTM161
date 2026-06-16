---
name: chrome-devtools-mcp
description: Drive the Chrome DevTools MCP against the local Coffee Shop Agent Observatory (or any localhost UI). Covers the WSL setup gotchas, the MCP config, and the right way to wait for a conversation run to finish. Use when the user wants to open a localhost URL in chrome via MCP, run a scenario in the observatory UI, or debug "Target closed" errors from the chrome-devtools MCP.
---

# Chrome DevTools MCP — runbook

## Environment notes (WSL2, this machine)

- The MCP server runs in WSL. It cannot drive Chrome on the Windows side.
- Default auto-discovery picks the **snap** Chromium, which fails to launch under MCP control ("Target closed" on every tool call). Do not rely on auto-discovery.
- A working Chrome is installed at:
  `/home/alorcus/.cache/puppeteer/chrome/linux-149.0.7827.54/chrome-linux64/chrome`
  (Chrome for Testing 149, fetched via `npx @puppeteer/browsers install chrome@stable --path ~/.cache/puppeteer`. Requires `unzip` on the system.)
- `~/.claude.json` is configured to point the MCP at that binary via `--executablePath`. If a fresh session shows "Target closed" on every call, check that section is still present:

  ```json
  "chrome-devtools": {
    "type": "stdio",
    "command": "npx",
    "args": [
      "chrome-devtools-mcp@latest",
      "--executablePath",
      "/home/alorcus/.cache/puppeteer/chrome/linux-149.0.7827.54/chrome-linux64/chrome"
    ],
    "env": {}
  }
  ```

  After editing, the user must run `/mcp` to reconnect — the MCP only re-reads config on reconnect.

## Healthy-state check

`mcp__chrome-devtools__list_pages` should return a list (often just `about:blank`). If it returns `Protocol error (Target.setDiscoverTargets): Target closed`, the MCP is connected but its browser process died — usually the snap fallback. Verify the `--executablePath` arg above, then ask the user to `/mcp` reconnect.

## App under test — Coffee Shop Agent Observatory

- URL: `http://localhost:5006/?theme=default`
- Open with: `mcp__chrome-devtools__new_page url=http://localhost:5006/?theme=default`
- The user starts the server themselves; do not try to start it from here.
- The page has: a Scenario dropdown (4 presets), a Log Level dropdown, an editable Customer Prompt, a "Run Conversation" button, a Conversation Log, and four agent panels (Order, Inventory, Barista, Customer Service) plus inventory + coffee machine state.

## Running a conversation end-to-end

1. `take_snapshot` to get current uids — the "Run Conversation" button is the one with `name="Run Conversation"`.
2. `click` that uid.
3. **Do not** use `wait_for` with the text `"DONE"` — the prompt template itself contains the word "DONE" and `wait_for` matches substrings, so it returns immediately. Same trap with `"complete"` (matches "Conversation complete" header before run starts) and short tokens like `"ready"`.
4. Instead poll until the **last** Customer line equals `DONE`:

   ```js
   // evaluate_script
   async () => {
     const deadline = Date.now() + 180000;
     while (Date.now() < deadline) {
       const text = document.body.innerText;
       // Find the LAST "Customer: ..." line in the conversation log
       const lines = text.split('\n').map(l => l.trim());
       const lastCustomer = [...lines].reverse().find(l => /^Customer\s*:/.test(l));
       if (lastCustomer && /^Customer\s*:\s*DONE\s*$/.test(lastCustomer)) {
         return { done: true };
       }
       await new Promise(r => setTimeout(r, 2000));
     }
     return { done: false, tail: document.body.innerText.slice(-2000) };
   }
   ```

   Conversations typically take 30–90 s. Budget 3 minutes.

## Reading agent state during/after a run

- The conversation transcript is rendered as repeated `HH:MM:SS  <speaker>:  <message>` lines under "Conversation Log".
- Each agent panel shows its own message history (User/AI exchanges) plus a status string: `idle` / `thinking` / `working`.
- Inventory tiles show `Stock: N` plus emoji repeats; the tray shows current items or `Empty`.
- The Coffee Machine panel shows a state pill (INIT / SUCC / FAIL etc.) and a status sentence like "Idle — waiting for orders".

## Common pitfalls

- **Stale uids after a run.** The conversation log re-renders new uids. Always `take_snapshot` again before clicking anything that appeared after the run started.
- **`wait_for` is dumb.** It does substring matching on the whole accessibility tree, including the prompt textbox value. Prefer `evaluate_script` polling for anything past trivial.
- **Don't `take_screenshot` to "see what's happening" mid-run** — the page is dense and the snapshot tree is more useful. Screenshot only when the user asks for one or when verifying visual styling.
- **The first MCP call after `/mcp` reconnect can be slow** (Chrome cold-start, ~1–2 s). Don't retry on the first call's apparent hang.

## Quick reset

To start a clean run after one finishes: `navigate_page type=reload` reloads `localhost:5006`, and the click sequence above starts a fresh conversation. The "♻️" button on the Coffee Machine panel resets just the machine state, not the conversation.
