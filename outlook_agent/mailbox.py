"""Mailbox backends.

`GraphMailbox` talks to Outlook / Microsoft 365 through Microsoft Graph.
`DemoMailbox` is an in-memory fake used for `--demo` mode and tests.

Both return plain, model-friendly dicts so the agent never sees raw Graph JSON.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import httpx

GRAPH = "https://graph.microsoft.com/v1.0"
MSG_FIELDS = (
    "id,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead,importance,"
    "hasAttachments,bodyPreview,webLink,flag,conversationId"
)


class MailboxError(RuntimeError):
    pass


class Mailbox(Protocol):
    def me(self) -> dict: ...
    def list_messages(self, folder: str = "inbox", top: int = 10, unread_only: bool = False,
                      since_hours: float | None = None) -> list[dict]: ...
    def get_message(self, message_id: str) -> dict: ...
    def search(self, query: str, top: int = 10) -> list[dict]: ...
    def list_folders(self) -> list[dict]: ...
    def mark_read(self, message_id: str, is_read: bool = True) -> None: ...
    def flag(self, message_id: str, flagged: bool = True) -> None: ...
    def move(self, message_id: str, folder: str) -> None: ...
    def create_reply_draft(self, message_id: str, body: str, reply_all: bool = False) -> dict: ...
    def send_draft(self, draft_id: str) -> None: ...
    def send_mail(self, to: list[str], subject: str, body: str, cc: list[str] | None = None) -> None: ...
    def list_events(self, days: int = 1) -> list[dict]: ...


def html_to_text(html: str) -> str:
    """Crude fallback for HTML bodies (Graph usually returns text when asked)."""
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def clean_body(text: str, limit: int) -> str:
    """Drop quoted reply chains and trim, so the model sees the new content."""
    cut_markers = [r"\n-{2,}\s*Original Message", r"\nFrom: .+\nSent: ", r"\nOn .+ wrote:\n"]
    for marker in cut_markers:
        m = re.search(marker, text)
        if m and text[: m.start()].strip():
            text = text[: m.start()]
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …[truncated]"
    return text


def _addr(entry: dict | None) -> str:
    if not entry:
        return ""
    ea = entry.get("emailAddress", {})
    name, address = ea.get("name", ""), ea.get("address", "")
    return f"{name} <{address}>" if name and name != address else address


def _normalize(m: dict, body_limit: int | None = None) -> dict:
    out = {
        "id": m["id"],
        "subject": m.get("subject") or "(no subject)",
        "from": _addr(m.get("from")),
        "to": [_addr(r) for r in m.get("toRecipients", [])],
        "received": m.get("receivedDateTime", ""),
        "is_read": m.get("isRead", False),
        "importance": m.get("importance", "normal"),
        "flagged": (m.get("flag") or {}).get("flagStatus") == "flagged",
        "has_attachments": m.get("hasAttachments", False),
        "preview": (m.get("bodyPreview") or "")[:300],
    }
    if m.get("ccRecipients"):
        out["cc"] = [_addr(r) for r in m["ccRecipients"]]
    if body_limit is not None and "body" in m:
        content = m["body"].get("content", "")
        if m["body"].get("contentType", "").lower() == "html":
            content = html_to_text(content)
        out["body"] = clean_body(content, body_limit)
        out["web_link"] = m.get("webLink", "")
    return out


WELL_KNOWN_FOLDERS = {
    "inbox": "inbox", "archive": "archive", "deleted": "deleteditems", "trash": "deleteditems",
    "deleteditems": "deleteditems", "junk": "junkemail", "junkemail": "junkemail", "spam": "junkemail",
    "sent": "sentitems", "sentitems": "sentitems", "drafts": "drafts",
}


class GraphMailbox:
    def __init__(self, token_provider, body_limit: int = 2500, client: httpx.Client | None = None):
        self._token = token_provider
        self.body_limit = body_limit
        self.http = client or httpx.Client(base_url=GRAPH, timeout=30)
        self._folders: dict[str, str] | None = None

    # -- plumbing ---------------------------------------------------------------
    def _request(self, method: str, path: str, **kw) -> Any:
        headers = kw.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._token()}"
        headers.setdefault("Prefer", 'outlook.body-content-type="text"')
        for attempt in range(4):
            resp = self.http.request(method, path, headers=headers, **kw)
            if resp.status_code in (429, 503, 504) and attempt < 3:
                time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
                continue
            break
        if resp.status_code >= 400:
            try:
                detail = resp.json()["error"]["message"]
            except Exception:
                detail = resp.text[:300]
            raise MailboxError(f"Graph {method} {path} failed ({resp.status_code}): {detail}")
        return resp.json() if resp.content else None

    def _folder_id(self, folder: str) -> str:
        key = folder.strip().lower()
        if key in WELL_KNOWN_FOLDERS:
            return WELL_KNOWN_FOLDERS[key]
        if self._folders is None:
            self.list_folders()
        assert self._folders is not None
        if key in self._folders:
            return self._folders[key]
        raise MailboxError(f"Folder '{folder}' not found. Known: {', '.join(sorted(self._folders))}")

    # -- reads ------------------------------------------------------------------
    def me(self) -> dict:
        data = self._request("GET", "/me", params={"$select": "displayName,mail,userPrincipalName"})
        return {"name": data.get("displayName"), "email": data.get("mail") or data.get("userPrincipalName")}

    def list_messages(self, folder="inbox", top=10, unread_only=False, since_hours=None) -> list[dict]:
        params: dict[str, Any] = {
            "$top": max(1, min(int(top), 50)),
            "$select": MSG_FIELDS,
            "$orderby": "receivedDateTime desc",
        }
        filters = []
        if since_hours or unread_only:
            # Graph requires $orderby properties to appear first in $filter.
            since = datetime.now(timezone.utc) - timedelta(hours=float(since_hours or 24 * 3650))
            filters.append(f"receivedDateTime ge {since.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if unread_only:
            filters.append("isRead eq false")
        if filters:
            params["$filter"] = " and ".join(filters)
        data = self._request("GET", f"/me/mailFolders/{self._folder_id(folder)}/messages", params=params)
        return [_normalize(m) for m in data.get("value", [])]

    def get_message(self, message_id: str) -> dict:
        data = self._request("GET", f"/me/messages/{message_id}",
                             params={"$select": MSG_FIELDS + ",body"})
        return _normalize(data, self.body_limit)

    def search(self, query: str, top=10) -> list[dict]:
        q = query.replace('"', "'")
        data = self._request("GET", "/me/messages",
                             params={"$search": f'"{q}"', "$top": max(1, min(int(top), 50)),
                                     "$select": MSG_FIELDS})
        return [_normalize(m) for m in data.get("value", [])]

    def list_folders(self) -> list[dict]:
        data = self._request("GET", "/me/mailFolders",
                             params={"$top": 100, "$select": "id,displayName,unreadItemCount,totalItemCount"})
        folders = data.get("value", [])
        self._folders = {f["displayName"].lower(): f["id"] for f in folders}
        return [{"name": f["displayName"], "unread": f.get("unreadItemCount", 0),
                 "total": f.get("totalItemCount", 0)} for f in folders]

    def list_events(self, days=1) -> list[dict]:
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=max(1, int(days)))
        data = self._request("GET", "/me/calendarView", params={
            "startDateTime": start.isoformat(), "endDateTime": end.isoformat(),
            "$orderby": "start/dateTime", "$top": 50,
            "$select": "subject,start,end,location,organizer,isOnlineMeeting,onlineMeetingUrl",
        }, headers={"Prefer": 'outlook.timezone="UTC"'})
        return [{
            "subject": e.get("subject"),
            "start_utc": e["start"]["dateTime"][:16],
            "end_utc": e["end"]["dateTime"][:16],
            "location": (e.get("location") or {}).get("displayName", ""),
            "organizer": _addr(e.get("organizer")),
            "online": e.get("isOnlineMeeting", False),
        } for e in data.get("value", [])]

    # -- writes -----------------------------------------------------------------
    def mark_read(self, message_id: str, is_read=True) -> None:
        self._request("PATCH", f"/me/messages/{message_id}", json={"isRead": bool(is_read)})

    def flag(self, message_id: str, flagged=True) -> None:
        status = "flagged" if flagged else "notFlagged"
        self._request("PATCH", f"/me/messages/{message_id}", json={"flag": {"flagStatus": status}})

    def move(self, message_id: str, folder: str) -> None:
        self._request("POST", f"/me/messages/{message_id}/move",
                      json={"destinationId": self._folder_id(folder)})

    def create_reply_draft(self, message_id: str, body: str, reply_all=False) -> dict:
        action = "createReplyAll" if reply_all else "createReply"
        draft = self._request("POST", f"/me/messages/{message_id}/{action}", json={"comment": body})
        return {"draft_id": draft["id"], "subject": draft.get("subject"),
                "to": [_addr(r) for r in draft.get("toRecipients", [])]}

    def send_draft(self, draft_id: str) -> None:
        self._request("POST", f"/me/messages/{draft_id}/send")

    def send_mail(self, to, subject, body, cc=None) -> None:
        def rcpts(xs):
            return [{"emailAddress": {"address": a.strip()}} for a in xs or [] if a.strip()]
        self._request("POST", "/me/sendMail", json={
            "message": {"subject": subject, "body": {"contentType": "Text", "content": body},
                        "toRecipients": rcpts(to), "ccRecipients": rcpts(cc)},
            "saveToSentItems": True,
        })


class DemoMailbox:
    """A fake inbox so you can try the agent without an Outlook account."""

    def __init__(self, body_limit: int = 2500):
        self.body_limit = body_limit
        now = datetime.now(timezone.utc)

        def ago(h):
            return (now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")

        seed = [
            ("Priya Shah", "priya@acme.com", "Q4 budget review — need your numbers by Thursday",
             "Hi,\n\nFinance needs every team's Q4 forecast by Thursday 5pm so we can lock the budget. "
             "Please send your headcount plan and the cloud spend estimate. If anything is above the "
             "approved envelope, flag it with a one-line justification.\n\nThanks,\nPriya", 2, "high"),
            ("GitHub", "noreply@github.com", "[orchestration-sdk] CI failed on main",
             "Run #482 failed: tests/test_scheduler.py::test_retry_backoff — AssertionError: expected 3 "
             "retries, got 2.", 3, "normal"),
            ("Marco Ruiz", "marco@bigclient.io", "Contract renewal & pricing question",
             "Hey, we're planning our renewal for next year. Could we get a quote for 250 seats instead of "
             "150? Also, is SSO included in the enterprise tier? We'd like to decide before the 15th.\n\nMarco",
             5, "normal"),
            ("LinkedIn", "news@linkedin.com", "You appeared in 12 searches this week",
             "See who's looking at your profile. Upgrade to Premium to unlock insights.", 9, "low"),
            ("Dana Lee", "dana@acme.com", "Lunch Friday?",
             "Want to grab lunch Friday around 12:30? New ramen place opened on 3rd street.", 20, "normal"),
            ("IT Helpdesk", "it@acme.com", "Action required: password expires in 3 days",
             "Your corporate password expires in 3 days. Change it at https://id.acme.com before it expires "
             "to avoid losing VPN access.", 26, "high"),
            ("Shopify Deals", "deals@shop-promos.example", "48-hour flash sale: 70% off!!!",
             "Don't miss out on our biggest sale of the year. Click now.", 30, "low"),
            ("Priya Shah", "priya@acme.com", "Re: offsite planning",
             "Venue is confirmed for Nov 20-21. Can you own the agenda for day 2?\n\n"
             "On Mon, someone wrote:\n> earlier thread text", 50, "normal"),
        ]
        self.messages: dict[str, dict] = {}
        self.folder_of: dict[str, str] = {}
        for i, (name, addr, subj, body, hours, imp) in enumerate(seed, 1):
            mid = f"msg-{i}"
            self.messages[mid] = {
                "id": mid, "subject": subj, "from": {"emailAddress": {"name": name, "address": addr}},
                "toRecipients": [{"emailAddress": {"name": "You", "address": "you@acme.com"}}],
                "receivedDateTime": ago(hours), "isRead": hours > 24, "importance": imp,
                "hasAttachments": False, "bodyPreview": body[:200], "flag": {"flagStatus": "notFlagged"},
                "body": {"contentType": "text", "content": body}, "webLink": f"https://outlook.example/{mid}",
            }
            self.folder_of[mid] = "inbox"
        self.folders = ["Inbox", "Archive", "Deleted Items", "Junk Email", "Sent Items", "Drafts", "Newsletters"]
        self.drafts: dict[str, dict] = {}
        self.sent: list[dict] = []

    def _get(self, mid: str) -> dict:
        if mid not in self.messages:
            raise MailboxError(f"No message with id '{mid}'")
        return self.messages[mid]

    def _canon_folder(self, folder: str) -> str:
        key = folder.strip().lower()
        alias = {"deleted": "deleted items", "trash": "deleted items", "deleteditems": "deleted items",
                 "junk": "junk email", "spam": "junk email", "sent": "sent items"}
        key = alias.get(key, key)
        for f in self.folders:
            if f.lower() == key:
                return f.lower()
        raise MailboxError(f"Folder '{folder}' not found. Known: {', '.join(self.folders)}")

    def me(self):
        return {"name": "Demo User", "email": "you@acme.com"}

    def list_messages(self, folder="inbox", top=10, unread_only=False, since_hours=None):
        folder = self._canon_folder(folder)
        cutoff = None
        if since_hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=float(since_hours))
        out = []
        for mid, m in sorted(self.messages.items(), key=lambda kv: kv[1]["receivedDateTime"], reverse=True):
            if self.folder_of[mid] != folder or (unread_only and m["isRead"]):
                continue
            if cutoff and datetime.fromisoformat(m["receivedDateTime"].replace("Z", "+00:00")) < cutoff:
                continue
            out.append(_normalize(m))
        return out[: int(top)]

    def get_message(self, message_id):
        return _normalize(self._get(message_id), self.body_limit)

    def search(self, query, top=10):
        terms = query.lower().split()
        hits = []
        for m in self.messages.values():
            hay = " ".join([m["subject"], m["body"]["content"], _addr(m["from"])]).lower()
            if all(t in hay for t in terms):
                hits.append(_normalize(m))
        return hits[: int(top)]

    def list_folders(self):
        return [{"name": f, "unread": sum(1 for mid, m in self.messages.items()
                                          if self.folder_of[mid] == f.lower() and not m["isRead"]),
                 "total": sum(1 for v in self.folder_of.values() if v == f.lower())} for f in self.folders]

    def list_events(self, days=1):
        today = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        return [
            {"subject": "Team standup", "start_utc": (today + timedelta(hours=1)).isoformat()[:16],
             "end_utc": (today + timedelta(hours=1, minutes=15)).isoformat()[:16], "location": "Teams",
             "organizer": "Priya Shah <priya@acme.com>", "online": True},
            {"subject": "1:1 with Marco (BigClient)", "start_utc": (today + timedelta(hours=4)).isoformat()[:16],
             "end_utc": (today + timedelta(hours=4, minutes=30)).isoformat()[:16], "location": "Room 4B",
             "organizer": "You <you@acme.com>", "online": False},
        ]

    def mark_read(self, message_id, is_read=True):
        self._get(message_id)["isRead"] = bool(is_read)

    def flag(self, message_id, flagged=True):
        self._get(message_id)["flag"] = {"flagStatus": "flagged" if flagged else "notFlagged"}

    def move(self, message_id, folder):
        self._get(message_id)
        self.folder_of[message_id] = self._canon_folder(folder)

    def create_reply_draft(self, message_id, body, reply_all=False):
        m = self._get(message_id)
        did = f"draft-{len(self.drafts) + 1}"
        self.drafts[did] = {"in_reply_to": message_id, "to": [_addr(m["from"])],
                            "subject": "RE: " + m["subject"], "body": body}
        return {"draft_id": did, "subject": self.drafts[did]["subject"], "to": self.drafts[did]["to"]}

    def send_draft(self, draft_id):
        if draft_id not in self.drafts:
            raise MailboxError(f"No draft with id '{draft_id}'")
        self.sent.append(self.drafts.pop(draft_id))

    def send_mail(self, to, subject, body, cc=None):
        self.sent.append({"to": list(to), "cc": list(cc or []), "subject": subject, "body": body})
