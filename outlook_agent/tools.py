"""Tools the agent can call, with JSON schemas for Ollama tool-calling.

Graph message ids are ~150 characters; an 8B model will mangle them. So the tools
layer hands the model short refs (E1, E2, … and D1, D2, … for drafts) and keeps
the mapping to real ids here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .llm import OllamaClient
from .mailbox import Mailbox, MailboxError
from . import summarize

ConfirmFn = Callable[[str], bool]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]
    needs_confirmation: bool = False

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


class RefMap:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.to_id: dict[str, str] = {}
        self.to_ref: dict[str, str] = {}

    def ref(self, real_id: str) -> str:
        if real_id not in self.to_ref:
            r = f"{self.prefix}{len(self.to_ref) + 1}"
            self.to_ref[real_id] = r
            self.to_id[r] = real_id
        return self.to_ref[real_id]

    def resolve(self, ref: str) -> str:
        key = str(ref).strip().upper().strip("[]")
        if key in self.to_id:
            return self.to_id[key]
        if ref in self.to_ref:  # model passed a real id
            return ref
        raise MailboxError(f"Unknown reference '{ref}'. List or search emails first to get refs like E1.")


class Toolbox:
    def __init__(self, mailbox: Mailbox, llm: OllamaClient, confirm: ConfirmFn,
                 progress: Callable[[str], None] | None = None):
        self.mailbox = mailbox
        self.llm = llm
        self.confirm = confirm
        self.progress = progress
        self.emails = RefMap("E")
        self.drafts = RefMap("D")
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

    # -- helpers ----------------------------------------------------------------
    def _brief(self, m: dict) -> dict:
        out = {"ref": self.emails.ref(m["id"]), "from": m["from"], "subject": m["subject"],
               "received": m["received"][:16].replace("T", " "), "unread": not m["is_read"],
               "preview": m["preview"][:160]}
        if m["importance"] == "high":
            out["important"] = True
        if m.get("flagged"):
            out["flagged"] = True
        return out

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool '{name}'. Available: {', '.join(self.tools)}"})
        try:
            if tool.needs_confirmation:
                if not self.confirm(self.describe(name, args)):
                    return json.dumps({"status": "cancelled", "note": "The user declined this action. "
                                       "Do not retry it unless they ask."})
            result = tool.handler(**args)
        except TypeError as e:
            return json.dumps({"error": f"Bad arguments for {name}: {e}"})
        except MailboxError as e:
            return json.dumps({"error": str(e)})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def describe(self, name: str, args: dict) -> str:
        """Human-readable description of a sensitive action, shown before confirming."""
        if name == "send_email":
            return (f"SEND EMAIL\n  To: {', '.join(args.get('to', []))}\n"
                    + (f"  Cc: {', '.join(args.get('cc') or [])}\n" if args.get("cc") else "")
                    + f"  Subject: {args.get('subject')}\n\n{args.get('body')}")
        if name == "send_draft":
            return f"SEND DRAFT {args.get('draft_ref')}"
        if name == "move_email":
            subj = self._subject_for(args.get("email_ref", ""))
            return f"MOVE {args.get('email_ref')} ({subj}) → {args.get('folder')}"
        return f"{name}({json.dumps(args)})"

    def _subject_for(self, ref: str) -> str:
        try:
            return self.mailbox.get_message(self.emails.resolve(ref))["subject"]
        except Exception:
            return "?"

    # -- tool implementations ---------------------------------------------------
    def list_emails(self, folder: str = "inbox", count: int = 10, unread_only: bool = False,
                    since_hours: float | None = None):
        msgs = self.mailbox.list_messages(folder, int(count), bool(unread_only), since_hours)
        return {"folder": folder, "count": len(msgs), "emails": [self._brief(m) for m in msgs]}

    def read_email(self, email_ref: str):
        m = self.mailbox.get_message(self.emails.resolve(email_ref))
        m["ref"] = self.emails.ref(m.pop("id"))
        return m

    def search_emails(self, query: str, count: int = 10):
        msgs = self.mailbox.search(query, int(count))
        return {"query": query, "count": len(msgs), "emails": [self._brief(m) for m in msgs]}

    def list_folders(self):
        return {"folders": self.mailbox.list_folders()}

    def get_calendar(self, days: int = 1):
        return {"days": days, "events": self.mailbox.list_events(int(days))}

    def summarize_inbox(self, count: int = 15, unread_only: bool = True, since_hours: float | None = None,
                        folder: str = "inbox"):
        msgs = self.mailbox.list_messages(folder, int(count), bool(unread_only), since_hours)
        if not msgs:
            return "No matching emails — the inbox is clear."
        items = summarize.triage(self.llm, self.mailbox, msgs, self.emails.ref, progress=self.progress)
        digest = summarize.render_digest(items, summarize.tldr(self.llm, items))
        return "Digest (show this to the user as-is, you may add a short comment):\n\n" + digest

    def mark_as_read(self, email_ref: str, read: bool = True):
        self.mailbox.mark_read(self.emails.resolve(email_ref), bool(read))
        return {"status": "ok", "ref": email_ref, "read": bool(read)}

    def flag_email(self, email_ref: str, flagged: bool = True):
        self.mailbox.flag(self.emails.resolve(email_ref), bool(flagged))
        return {"status": "ok", "ref": email_ref, "flagged": bool(flagged)}

    def move_email(self, email_ref: str, folder: str):
        self.mailbox.move(self.emails.resolve(email_ref), folder)
        return {"status": "moved", "ref": email_ref, "folder": folder}

    def draft_reply(self, email_ref: str, body: str, reply_all: bool = False):
        d = self.mailbox.create_reply_draft(self.emails.resolve(email_ref), body, bool(reply_all))
        ref = self.drafts.ref(d.pop("draft_id"))
        return {"status": "draft saved in Drafts folder (not sent)", "draft_ref": ref, **d, "body": body}

    def send_draft(self, draft_ref: str):
        self.mailbox.send_draft(self.drafts.resolve(draft_ref))
        return {"status": "sent", "draft_ref": draft_ref}

    def send_email(self, to: list[str] | str, subject: str, body: str, cc: list[str] | str | None = None):
        def as_list(x):
            if not x:
                return []
            return [a.strip() for a in (x.split(",") if isinstance(x, str) else x) if a.strip()]
        self.mailbox.send_mail(as_list(to), subject, body, as_list(cc))
        return {"status": "sent", "to": as_list(to), "subject": subject}

    # -- registry ---------------------------------------------------------------
    def _build(self) -> list[Tool]:
        ref = {"type": "string", "description": "Short email ref like 'E3' from a previous list/search"}
        return [
            Tool("summarize_inbox",
                 "Triage and summarize recent emails into a prioritized digest with suggested actions. "
                 "Use this for 'summarize my inbox', 'what did I miss', 'what needs my attention'.",
                 _obj({"count": {"type": "integer", "description": "How many recent emails (default 15, max 50)"},
                       "unread_only": {"type": "boolean", "description": "Only unread emails (default true)"},
                       "since_hours": {"type": "number", "description": "Only emails from the last N hours"},
                       "folder": {"type": "string", "description": "Folder name, default inbox"}}),
                 self.summarize_inbox),
            Tool("list_emails", "List recent emails (subject, sender, preview) in a folder.",
                 _obj({"folder": {"type": "string", "description": "inbox (default), archive, sent, junk, or a folder name"},
                       "count": {"type": "integer", "description": "Max emails, default 10"},
                       "unread_only": {"type": "boolean"},
                       "since_hours": {"type": "number", "description": "Only emails from the last N hours"}}),
                 self.list_emails),
            Tool("read_email", "Read the full body of one email.", _obj({"email_ref": ref}, ["email_ref"]),
                 self.read_email),
            Tool("search_emails", "Search all mail by keywords, sender name/address, or subject.",
                 _obj({"query": {"type": "string"}, "count": {"type": "integer"}}, ["query"]),
                 self.search_emails),
            Tool("list_folders", "List mail folders with unread counts.", _obj({}), self.list_folders),
            Tool("get_calendar", "Get upcoming calendar events (times are UTC).",
                 _obj({"days": {"type": "integer", "description": "How many days ahead, default 1"}}),
                 self.get_calendar),
            Tool("mark_as_read", "Mark an email read (or unread with read=false).",
                 _obj({"email_ref": ref, "read": {"type": "boolean"}}, ["email_ref"]), self.mark_as_read),
            Tool("flag_email", "Flag an email for follow-up (or unflag with flagged=false).",
                 _obj({"email_ref": ref, "flagged": {"type": "boolean"}}, ["email_ref"]), self.flag_email),
            Tool("move_email", "Move an email to a folder (archive, deleted, junk, or a folder name).",
                 _obj({"email_ref": ref, "folder": {"type": "string"}}, ["email_ref", "folder"]),
                 self.move_email, needs_confirmation=True),
            Tool("draft_reply", "Write a reply and save it as a draft (does NOT send). "
                 "Body is plain text written in the user's voice; do not include the quoted original.",
                 _obj({"email_ref": ref, "body": {"type": "string"},
                       "reply_all": {"type": "boolean"}}, ["email_ref", "body"]),
                 self.draft_reply),
            Tool("send_draft", "Send a previously created draft. Only when the user explicitly asks to send.",
                 _obj({"draft_ref": {"type": "string", "description": "Draft ref like 'D1'"}}, ["draft_ref"]),
                 self.send_draft, needs_confirmation=True),
            Tool("send_email", "Compose and send a new email. Only when the user explicitly asks to send.",
                 _obj({"to": {"type": "array", "items": {"type": "string"}, "description": "Recipient addresses"},
                       "subject": {"type": "string"}, "body": {"type": "string"},
                       "cc": {"type": "array", "items": {"type": "string"}}}, ["to", "subject", "body"]),
                 self.send_email, needs_confirmation=True),
        ]
