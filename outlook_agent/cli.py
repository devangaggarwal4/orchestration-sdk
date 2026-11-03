"""Command-line entry point.

  outlook-agent chat              interactive agent (default)
  outlook-agent digest            one-shot prioritized inbox digest
  outlook-agent watch             poll for new mail and alert on important messages
  outlook-agent login / logout    manage the Microsoft sign-in

Add --demo to any command to use a built-in fake mailbox instead of Outlook.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from .agent import Agent
from .config import Settings
from .llm import LLMError, OllamaClient
from .mailbox import DemoMailbox, GraphMailbox, MailboxError
from .tools import Toolbox
from . import summarize

console = Console()


def build(settings: Settings, demo: bool):
    if demo:
        return DemoMailbox(settings.max_body_chars)
    from .auth import GraphAuth
    auth = GraphAuth(settings)
    auth.token()  # sign in up front so the prompt doesn't appear mid-conversation
    return GraphMailbox(auth.token, settings.max_body_chars)


def make_llm(settings: Settings, model: str | None) -> OllamaClient:
    llm = OllamaClient(settings.ollama_host, model or settings.ollama_model,
                       settings.ollama_num_ctx, settings.ollama_think)
    llm.check()
    return llm


def confirm(description: str) -> bool:
    console.print(Panel(description, title="Confirm action", border_style="yellow"))
    return Confirm.ask("Proceed?", default=False)


def progress(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


def on_tool(name: str, args: dict) -> None:
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "body")
    console.print(f"[dim cyan]→ {name}({shown})[/dim cyan]")


def cmd_chat(settings: Settings, args) -> None:
    mailbox = build(settings, args.demo)
    llm = make_llm(settings, args.model)
    agent = Agent(llm, Toolbox(mailbox, llm, confirm, progress), on_tool=on_tool)
    console.print(Panel.fit(
        f"[bold]Outlook Agent[/bold] · model [cyan]{llm.model}[/cyan]"
        + (" · [magenta]demo mailbox[/magenta]" if args.demo else "")
        + "\nTry: [italic]summarize my unread email[/italic] · [italic]what's on my calendar today?[/italic] · "
          "[italic]draft a reply to E1 saying I'll send numbers tomorrow[/italic]\n"
          "Commands: /reset  /quit", border_style="blue"))
    while True:
        try:
            text = console.input("[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "quit", "exit"):
            break
        if text == "/reset":
            agent.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        try:
            with console.status("thinking…", spinner="dots"):
                answer = agent.run(text)
        except (LLMError, MailboxError) as e:
            console.print(f"[red]Error:[/red] {e}")
            continue
        console.print(Markdown(answer))
        console.print()


def cmd_digest(settings: Settings, args) -> None:
    mailbox = build(settings, args.demo)
    llm = make_llm(settings, args.model)
    toolbox = Toolbox(mailbox, llm, confirm, progress)
    result = toolbox.summarize_inbox(count=args.count, unread_only=not args.all,
                                     since_hours=args.since_hours, folder=args.folder)
    digest = result.split("\n\n", 1)[-1]
    console.print(Markdown(digest))
    if args.output:
        Path(args.output).write_text(digest + "\n")
        console.print(f"[dim]Saved to {args.output}[/dim]")


def cmd_watch(settings: Settings, args) -> None:
    mailbox = build(settings, args.demo)
    llm = make_llm(settings, args.model)
    toolbox = Toolbox(mailbox, llm, confirm)
    seen: set[str] = set()
    first = True
    console.print(f"[blue]Watching inbox every {args.interval} min. Ctrl+C to stop.[/blue]")
    try:
        while True:
            msgs = [m for m in mailbox.list_messages("inbox", 25, unread_only=True) if m["id"] not in seen]
            seen.update(m["id"] for m in msgs)
            if first and not args.include_existing:
                msgs, first = [], False
            if msgs:
                items = summarize.triage(llm, mailbox, msgs, toolbox.emails.ref)
                stamp = datetime.now().strftime("%H:%M")
                for t in items:
                    if t.priority == "high" or args.verbose:
                        color = "red" if t.priority == "high" else "white"
                        console.print(f"[{color}]{stamp} [{t.priority.upper()}] {t.subject}[/{color}] — "
                                      f"{t.sender}\n   {t.summary}" + (f"\n   ➜ {t.action}" if t.action else ""))
                        if t.priority == "high":
                            console.bell()
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


def cmd_login(settings: Settings, args) -> None:
    from .auth import GraphAuth
    auth = GraphAuth(settings)
    me = GraphMailbox(auth.token).me()
    console.print(f"[green]Signed in as {me['name']} <{me['email']}>[/green]")


def cmd_logout(settings: Settings, args) -> None:
    from .auth import GraphAuth
    GraphAuth(settings).sign_out()
    console.print("Signed out and cleared token cache.")


def main(argv: list[str] | None = None) -> None:
    def common_opts(defaults: bool) -> argparse.ArgumentParser:
        # Subcommands use SUPPRESS so `--demo chat` and `chat --demo` both work.
        p = argparse.ArgumentParser(add_help=False)
        kw = {} if defaults else {"default": argparse.SUPPRESS}
        p.add_argument("--demo", action="store_true", help="use a fake mailbox (no Outlook needed)", **kw)
        p.add_argument("--model", help="Ollama model (default from OLLAMA_MODEL or qwen3:8b)", **kw)
        return p

    common = common_opts(False)
    parser = argparse.ArgumentParser(prog="outlook-agent", parents=[common_opts(True)],
                                     description="Local AI agent for your Outlook inbox (Ollama).")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("chat", parents=[common], help="interactive agent (default)")
    d = sub.add_parser("digest", parents=[common], help="one-shot inbox digest")
    d.add_argument("--count", type=int, default=20)
    d.add_argument("--all", action="store_true", help="include read emails")
    d.add_argument("--since-hours", type=float)
    d.add_argument("--folder", default="inbox")
    d.add_argument("-o", "--output", help="also write the digest to a markdown file")
    w = sub.add_parser("watch", parents=[common], help="alert on important new mail")
    w.add_argument("--interval", type=float, default=5, help="minutes between checks")
    w.add_argument("--include-existing", action="store_true", help="also triage mail already unread at start")
    w.add_argument("-v", "--verbose", action="store_true", help="show all new mail, not just high priority")
    sub.add_parser("login", help="sign in to Microsoft")
    sub.add_parser("logout", help="sign out")

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    handlers = {None: cmd_chat, "chat": cmd_chat, "digest": cmd_digest, "watch": cmd_watch,
                "login": cmd_login, "logout": cmd_logout}
    try:
        handlers[args.cmd](settings, args)
    except (LLMError, MailboxError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
