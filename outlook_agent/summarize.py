"""Batch triage + digest pipeline.

Small local models do much better summarizing a few emails at a time than a whole
inbox in one prompt, so we triage in batches (JSON output) and assemble the digest
deterministically, with one final short LLM call for the TL;DR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .llm import OllamaClient
from .mailbox import Mailbox

PRIORITIES = ("high", "medium", "low")
CATEGORIES = ("action", "reply", "meeting", "fyi", "newsletter", "spam")

TRIAGE_PROMPT = """You triage emails for a busy professional. For EACH email below return an object:
- "ref": the email's ref exactly as given
- "summary": one sentence (max 25 words) with the key point, names, dates and asks
- "priority": "high" (deadline/boss/client/security, needs attention soon), "medium", or "low" (promo, social, automated FYI)
- "category": one of action, reply, meeting, fyi, newsletter, spam
- "action": a short suggested next step for the user, or "" if none

Respond ONLY with JSON: {"emails": [ ... ]}

EMAILS:
"""


@dataclass
class Triaged:
    ref: str
    subject: str
    sender: str
    received: str
    summary: str
    priority: str
    category: str
    action: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _format_for_prompt(ref: str, msg: dict) -> str:
    body = msg.get("body") or msg.get("preview", "")
    return (f"--- ref: {ref}\nFrom: {msg['from']}\nReceived: {msg['received']}\n"
            f"Importance: {msg['importance']}\nSubject: {msg['subject']}\n\n{body[:900]}\n")


def triage(llm: OllamaClient, mailbox: Mailbox, messages: list[dict], ref_of: Callable[[str], str],
           batch_size: int = 4, progress: Callable[[str], None] | None = None) -> list[Triaged]:
    results: list[Triaged] = []
    for start in range(0, len(messages), batch_size):
        batch = messages[start:start + batch_size]
        if progress:
            progress(f"Reading emails {start + 1}-{start + len(batch)} of {len(messages)}…")
        full = []
        for m in batch:
            try:
                full.append(mailbox.get_message(m["id"]))
            except Exception:
                full.append(m)
        refs = [ref_of(m["id"]) for m in full]
        prompt = TRIAGE_PROMPT + "\n".join(_format_for_prompt(r, m) for r, m in zip(refs, full))
        parsed: dict[str, dict] = {}
        try:
            reply = llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            data = json.loads(reply.content)
            items = data.get("emails", data if isinstance(data, list) else [])
            parsed = {str(it.get("ref", "")).strip(): it for it in items if isinstance(it, dict)}
        except (json.JSONDecodeError, AttributeError):
            pass
        for ref, m in zip(refs, full):
            it = parsed.get(ref, {})
            prio = str(it.get("priority", "medium")).lower()
            cat = str(it.get("category", "fyi")).lower()
            results.append(Triaged(
                ref=ref, subject=m["subject"], sender=m["from"], received=m["received"],
                summary=str(it.get("summary") or m.get("preview", "")[:160]),
                priority=prio if prio in PRIORITIES else "medium",
                category=cat if cat in CATEGORIES else "fyi",
                action=str(it.get("action") or ""),
            ))
    return results


def tldr(llm: OllamaClient, items: list[Triaged]) -> str:
    if not items:
        return "Nothing new."
    lines = "\n".join(f"- [{t.priority}] {t.sender.split('<')[0].strip()}: {t.summary}" for t in items)
    reply = llm.chat([{"role": "user", "content":
        "Write a 2-3 sentence TL;DR of this inbox for its owner. Lead with what needs action "
        "and any deadlines. Plain text, no preamble.\n\n" + lines}])
    return reply.content


def render_digest(items: list[Triaged], overview: str) -> str:
    """Markdown digest grouped by priority."""
    out = ["## Inbox digest", "", overview.strip(), ""]
    for prio, title in (("high", "🔴 Needs attention"), ("medium", "🟡 Worth a look"), ("low", "⚪ Low priority")):
        group = [t for t in items if t.priority == prio]
        if not group:
            continue
        out.append(f"### {title} ({len(group)})")
        for t in group:
            who = t.sender.split("<")[0].strip() or t.sender
            line = f"- **[{t.ref}] {t.subject}** — {who}: {t.summary}"
            if t.action:
                line += f"  \n  ➜ _{t.action}_"
            out.append(line)
        out.append("")
    return "\n".join(out).strip()
