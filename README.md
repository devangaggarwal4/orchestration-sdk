# Outlook Agent

A local AI agent for your Outlook / Microsoft 365 mailbox. It runs **Qwen3-8B through Ollama**, so the model runs on your own machine. It connects to Outlook through **Microsoft Graph**.

What it can do:

- **Summarize and triage** your inbox into a digest sorted by priority, with a suggested next step for each email
- **Chat** with your mailbox: "what did Marco ask about?", "anything from finance this week?"
- **Read, search, flag, mark read, and move** emails
- **Draft replies** that are based on the actual email. Drafts are saved to your Drafts folder.
- **Send** emails and drafts. The agent always asks you before it sends anything or moves an email.
- **Check your calendar**
- **Watch mode**: checks for new mail on a timer and alerts you when an important email arrives

```
you › summarize my unread email from today
→ summarize_inbox(since_hours=24)
## Inbox digest
Priya needs your Q4 headcount and cloud spend by Thursday 5pm; Marco (BigClient) wants a 250-seat quote before the 15th…

### 🔴 Needs attention (3)
- [E1] Q4 budget review — Priya Shah: Q4 forecast due Thursday 5pm…
  ➜ Send headcount plan and cloud spend estimate
…
you › draft a reply to E3 saying SSO is included and I'll send a quote tomorrow
→ read_email(email_ref='E3')
→ draft_reply(email_ref='E3', reply_all=False)
Draft D1 saved…
you › send it
╭─ Confirm action ─╮
│ SEND DRAFT D1    │
╰──────────────────╯
Proceed? [y/N]
```

## Quick start

### 1. Install Ollama and the model

```bash
ollama pull qwen3:8b
```

The model needs about 5–6 GB of RAM or VRAM. Ollama must be running (`ollama serve`, or the desktop app).

### 2. Install the agent

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e .
```

### 3. Try it without Outlook

```bash
outlook-agent --demo
```

This uses a built-in fake inbox, so you can see how the agent works before you connect your account.

### 4. Connect your Outlook account (one-time, about 5 minutes)

The agent signs in as you with the **device-code flow**: you open a link, enter a code, and approve. You'll need your own (free) Azure app registration:

1. Go to [Azure portal → App registrations](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade) → **New registration**.
   - Name: `Outlook Agent`
   - Supported account types: **"Accounts in any organizational directory and personal Microsoft accounts"**. Use this for outlook.com/hotmail accounts. For work accounts only, pick single-tenant.
   - Redirect URI: leave empty.
2. On the app's **Authentication** page, set **Allow public client flows** to **Yes** and save.
3. On the **API permissions** page, click **Add a permission** → **Microsoft Graph** → **Delegated**, then add `Mail.ReadWrite`, `Mail.Send`, `Calendars.Read` and `User.Read`.
   - Work or school account: your tenant admin may need to grant consent.
4. Copy the **Application (client) ID** from the Overview page.

Then:

```bash
cp .env.example .env
```

Put the client ID in `.env` as `OUTLOOK_CLIENT_ID=<your id>`. If you registered a single-tenant app, also set `OUTLOOK_TENANT` to your tenant ID.

```bash
outlook-agent login
```

The token is cached in `~/.outlook_agent/token_cache.json` (permissions `0600`), so you only sign in once.

## Usage

| Command | What it does |
|---|---|
| `outlook-agent` / `outlook-agent chat` | Interactive agent. `/reset` clears the conversation, `/quit` exits. |
| `outlook-agent digest` | One-shot digest of unread mail. Options: `--all`, `--since-hours 24`, `--count 30`, `-o digest.md` |
| `outlook-agent watch` | Checks the inbox every 5 minutes and alerts on high-priority mail. Options: `--interval 10`, `-v` |
| `outlook-agent logout` | Clears the cached sign-in |

Every command accepts `--demo` and `--model <ollama-model>`.

Settings go in `.env` (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `qwen3:8b` | Any Ollama model that supports tool calling |
| `OLLAMA_NUM_CTX` | `8192` | Raise it if you have the memory. Long threads benefit. |
| `OLLAMA_THINK` | `0` | `1` turns on qwen3's reasoning mode: slower, sometimes better at multi-step requests |
| `OUTLOOK_MAX_BODY_CHARS` | `2500` | How much of each email body the model sees |

## How it works

```
 you ──► Agent loop (agent.py) ──► Ollama /api/chat (qwen3:8b, tools=[…])
              ▲    │ tool calls
              │    ▼
         Toolbox (tools.py) ── confirm() for send/move
              │
              ▼
      Mailbox (mailbox.py) ──► Microsoft Graph  (or DemoMailbox)
```

- **Agent loop**: the model either answers or calls tools. Tool results are sent back to the model, and the loop repeats for up to 8 steps. If the model repeats an identical call, the agent tells it to use the earlier result instead of running the call again.
- **Short refs**: Graph message IDs are about 150 characters long, and an 8B model will garble them. The model sees `E1`, `E2`… and `D1`… instead. The toolbox maps them back to the real IDs.
- **Summaries in batches**: `summarize_inbox` triages 4 emails per call in JSON mode. The digest is assembled in code and grouped by priority, with one final short LLM call for the TL;DR. A small model handles this much more reliably than a whole inbox in one prompt.
- **Body cleanup**: Graph is asked for plain-text bodies. Quoted reply chains are removed and long bodies are truncated, which keeps the context small.
- **Safety**: `send_email`, `send_draft` and `move_email` always show you the details and wait for a yes. `draft_reply` only saves a draft. Nothing is deleted outright: "delete" means moving the email to Deleted Items, which also asks for confirmation.

Code layout: `outlook_agent/{config,auth,mailbox,llm,tools,summarize,agent,cli}.py`.

## Development

```bash
pip install -e '.[dev]'
```

```bash
pytest
```

The tests use the demo mailbox, a scripted fake LLM, and mocked HTTP for Graph and Ollama. They don't need Ollama or an Outlook account.

## Ideas to extend

- Run `digest -o` on a schedule (cron) and email the digest to yourself with `send_email`
- Add a `create_event` tool (Graph `POST /me/events`) so the agent can accept meeting requests
- Auto-file newsletters into a folder during `watch`
- Swap in a larger model (`--model qwen3:14b`) for better reply drafting
