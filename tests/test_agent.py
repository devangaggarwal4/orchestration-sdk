import json

import httpx
import pytest

from outlook_agent.agent import Agent
from outlook_agent.llm import LLMReply, OllamaClient, ToolCall, strip_think
from outlook_agent.mailbox import DemoMailbox, GraphMailbox, clean_body
from outlook_agent.tools import Toolbox


class ScriptedLLM:
    """Stands in for Ollama: returns queued replies, or triage JSON when json_mode is used."""

    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls.append(messages)
        if json_mode:
            prompt = messages[-1]["content"]
            refs = [ln.split("ref: ")[1] for ln in prompt.splitlines() if ln.startswith("--- ref: ")]
            items = [{"ref": r, "summary": f"summary of {r}", "priority": "high" if r == "E1" else "low",
                      "category": "action", "action": "do it" if r == "E1" else ""} for r in refs]
            return LLMReply(content=json.dumps({"emails": items}))
        return self.replies.pop(0)


def make(replies, confirm=lambda _: True):
    mb = DemoMailbox()
    llm = ScriptedLLM(replies)
    tb = Toolbox(mb, llm, confirm)
    return mb, llm, tb, Agent(llm, tb)


def test_list_then_answer_uses_short_refs():
    mb, llm, tb, agent = make([
        LLMReply("", [ToolCall("list_emails", {"count": 3})]),
        LLMReply("You have 3 emails."),
    ])
    assert agent.run("what's in my inbox?") == "You have 3 emails."
    tool_msg = next(m for m in agent.history if m["role"] == "tool")
    data = json.loads(tool_msg["content"])
    assert [e["ref"] for e in data["emails"]] == ["E1", "E2", "E3"]
    assert tb.emails.resolve("e1") == "msg-1"


def test_draft_and_send_requires_confirmation():
    prompts = []
    mb, llm, tb, agent = make([
        LLMReply("", [ToolCall("list_emails", {})]),
        LLMReply("", [ToolCall("draft_reply", {"email_ref": "E1", "body": "Will send Thursday."})]),
        LLMReply("", [ToolCall("send_draft", {"draft_ref": "D1"})]),
        LLMReply("Sent."),
    ], confirm=lambda d: prompts.append(d) or True)
    agent.run("reply to priya and send it")
    assert len(prompts) == 1 and "SEND DRAFT D1" in prompts[0]
    assert mb.sent[0]["body"] == "Will send Thursday."


def test_declined_action_is_not_performed():
    mb, llm, tb, agent = make([
        LLMReply("", [ToolCall("send_email", {"to": ["a@b.c"], "subject": "hi", "body": "x"})]),
        LLMReply("OK, not sent."),
    ], confirm=lambda _: False)
    agent.run("email a@b.c")
    assert mb.sent == []
    assert "cancelled" in agent.history[-2]["content"]


def test_move_and_bad_ref_errors_are_reported_to_model():
    mb, llm, tb, agent = make([
        LLMReply("", [ToolCall("move_email", {"email_ref": "E9", "folder": "archive"})]),
        LLMReply("Couldn't find it."),
    ])
    agent.run("archive it")
    assert "Unknown reference" in agent.history[-2]["content"]
    tb.list_emails()
    tb.call("move_email", {"email_ref": "E4", "folder": "archive"})
    assert mb.folder_of["msg-4"] == "archive"


def test_summarize_inbox_digest_groups_by_priority():
    mb, llm, tb, _ = make([LLMReply("Priya needs Q4 numbers by Thursday.")])
    out = tb.summarize_inbox(count=5, unread_only=True)
    assert "Needs attention (1)" in out and "Low priority" in out
    assert "[E1]" in out and "Priya needs Q4 numbers" in out


def test_step_limit_forces_final_answer():
    loop = [LLMReply("", [ToolCall("list_folders", {})]) for _ in range(8)]
    mb, llm, tb, agent = make(loop + [LLMReply("Here are your folders.")])
    assert agent.run("folders?") == "Here are your folders."


def test_strip_think_and_clean_body():
    assert strip_think("<think>hmm</think>\nHello") == "Hello"
    body = "Sounds good, see you then." + "\n" * 2 + "On Mon, Bob wrote:\n> old stuff\n> more"
    assert clean_body(body, 1000) == "Sounds good, see you then."
    assert clean_body("x" * 50, 10).endswith("[truncated]")


def test_ollama_client_payload_and_tool_parsing():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "<think></think>",
                              "tool_calls": [{"function": {"name": "list_emails", "arguments": {"count": 2}}}]}})

    c = OllamaClient("http://x", "qwen3:8b")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    r = c.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert seen["model"] == "qwen3:8b" and seen["think"] is False and seen["stream"] is False
    assert r.tool_calls[0].name == "list_emails" and r.tool_calls[0].arguments == {"count": 2}
    assert r.content == ""


def test_graph_mailbox_requests():
    reqs = []

    def handler(request):
        reqs.append(request)
        if request.url.path.endswith("/messages") and request.method == "GET":
            return httpx.Response(200, json={"value": [{
                "id": "AAMk-long", "subject": "Hi", "from": {"emailAddress": {"name": "A", "address": "a@x"}},
                "receivedDateTime": "2026-11-01T10:00:00Z", "isRead": False, "importance": "normal",
                "bodyPreview": "hello"}]})
        return httpx.Response(202)

    mb = GraphMailbox(lambda: "tok", client=httpx.Client(base_url="https://graph.test",
                                                         transport=httpx.MockTransport(handler)))
    msgs = mb.list_messages(unread_only=True, top=5)
    assert msgs[0]["from"] == "A <a@x>"
    q = reqs[0].url.params
    assert q["$filter"].startswith("receivedDateTime ge") and "isRead eq false" in q["$filter"]
    assert reqs[0].headers["Authorization"] == "Bearer tok"
    mb.move("AAMk-long", "archive")
    assert json.loads(reqs[-1].content) == {"destinationId": "archive"}
    mb.send_mail(["b@x"], "S", "B")
    sent = json.loads(reqs[-1].content)["message"]
    assert sent["toRecipients"][0]["emailAddress"]["address"] == "b@x"
