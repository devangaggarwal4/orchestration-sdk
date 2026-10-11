import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, post, stream } from "../api/client";
import type { ChatEvent, ContentBlock, Conversation, StoredMessage } from "../api/types";
import { RouteCard } from "../components/RouteCard";
import { StepTimeline, ToolCallView } from "../components/StepTimeline";
import { Button, Empty, ErrorNote, Select, cx } from "../components/ui";
import { timeAgo } from "../lib/format";
import { emptyTimeline, reduceTimeline, type Timeline } from "../lib/timeline";

const TASK_TYPES = [
  "chat",
  "classify",
  "extract",
  "summarize",
  "rewrite",
  "code",
  "code_review",
  "plan",
  "multi_step_reasoning",
  "tool_use",
];

interface Settings {
  strategy: string;
  task_type: string;
  force_tier: string;
  tools: boolean;
}
const DEFAULTS: Settings = { strategy: "", task_type: "", force_tier: "", tools: false };

function blockText(b: ContentBlock): string {
  if (typeof b.content === "string") return b.content;
  if (Array.isArray(b.content)) return b.content.map((c) => c.text ?? "").join("");
  return b.text ?? "";
}

function StoredTurn({ m, results }: { m: StoredMessage; results: Map<string, ContentBlock> }) {
  const text = m.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  if (m.role === "user") {
    if (!text) return null; // tool results render under their tool call
    return (
      <div className="ml-auto max-w-[80%] rounded-xl bg-surface-2 px-3.5 py-2 text-sm whitespace-pre-wrap">
        {text}
      </div>
    );
  }
  const calls = m.content.filter((b) => b.type === "tool_use");
  const d = m.decisions[0];
  return (
    <div className="max-w-[90%]">
      {text && <p className="text-sm whitespace-pre-wrap text-ink">{text}</p>}
      {calls.length > 0 && (
        <div className="mt-2 space-y-1.5">
          {calls.map((c) => {
            const r = results.get(c.id ?? "");
            return (
              <ToolCallView
                key={c.id}
                name={c.name ?? "?"}
                input={c.input ?? {}}
                result={r ? blockText(r) : undefined}
                isError={r?.is_error}
              />
            );
          })}
        </div>
      )}
      {d && <RouteCard decision={d} costUsd={d.cost_usd} />}
    </div>
  );
}

export function ChatPage() {
  const { id } = useParams();
  // Remount per conversation so per-conversation state (settings edits) resets.
  return <ChatView key={id ?? "new"} id={id} />;
}

function ChatView({ id }: { id?: string }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [input, setInput] = useState("");
  const [overrides, setOverrides] = useState<Partial<Settings>>({});
  const [live, setLive] = useState<{ user: string; timeline: Timeline } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const list = useQuery({ queryKey: ["conversations"], queryFn: () => api<Conversation[]>("/api/conversations") });
  const conv = useQuery({
    queryKey: ["conversation", id],
    queryFn: () => api<Conversation>(`/api/conversations/${id}`),
    enabled: !!id,
  });

  const settings: Settings = {
    ...DEFAULTS,
    ...(conv.data?.settings as Partial<Settings> | undefined),
    ...overrides,
  };
  const setSettings = (s: Settings) => setOverrides(s);

  useEffect(() => bottom.current?.scrollIntoView({ block: "end" }), [conv.data, live]);

  const remove = useMutation({
    mutationFn: (cid: string) => api(`/api/conversations/${cid}`, { method: "DELETE" }),
    onSuccess: (_, cid) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      if (cid === id) navigate("/chat");
    },
  });

  const send = async () => {
    const content = input.trim();
    if (!content || live) return;
    setError(null);
    let cid = id;
    try {
      if (!cid) {
        // Navigate after the first reply: changing the URL remounts this view.
        const created = await post<Conversation>("/api/conversations", { settings });
        cid = created.id;
      } else {
        await api(`/api/conversations/${cid}`, { method: "PATCH", body: JSON.stringify({ settings }) });
      }
      setInput("");
      setLive({ user: content, timeline: emptyTimeline() });
      abort.current = new AbortController();
      await stream<ChatEvent>(
        `/api/conversations/${cid}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            content,
            strategy: settings.strategy || null,
            task_type: settings.task_type || null,
            force_tier: settings.force_tier || null,
            tools: settings.tools,
          }),
        },
        (ev) => setLive((l) => (l ? { ...l, timeline: reduceTimeline(l.timeline, ev) } : l)),
        abort.current.signal,
      );
      await qc.invalidateQueries({ queryKey: ["conversation", cid] });
      qc.invalidateQueries({ queryKey: ["conversations"] });
      if (cid !== id) navigate(`/chat/${cid}`, { replace: true });
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(e);
    } finally {
      setLive(null);
      abort.current = null;
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  const messages = conv.data?.messages ?? [];
  const results = new Map<string, ContentBlock>();
  for (const m of messages)
    for (const b of m.content) if (b.type === "tool_result" && b.tool_use_id) results.set(b.tool_use_id, b);

  return (
    <div className="grid gap-4 md:grid-cols-[240px_1fr]">
      <aside className="min-w-0 space-y-2">
        <Button variant="primary" className="w-full" onClick={() => navigate("/chat")}>
          New chat
        </Button>
        <ul className="space-y-0.5">
          {list.data?.map((c) => (
            <li key={c.id} className="group flex items-center">
              <button
                onClick={() => navigate(`/chat/${c.id}`)}
                className={cx(
                  "flex-1 truncate rounded-md px-2 py-1.5 text-left text-sm",
                  c.id === id ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface-2",
                )}
              >
                {c.title}
                <span className="block text-xs text-muted">{timeAgo(c.updated)}</span>
              </button>
              <button
                aria-label={`Delete ${c.title}`}
                onClick={() => remove.mutate(c.id)}
                className="invisible px-2 text-muted group-hover:visible hover:text-ink"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="flex min-h-[70vh] min-w-0 flex-col rounded-xl border border-line bg-surface">
        <div className="flex-1 space-y-4 overflow-y-auto p-4" data-testid="messages">
          {!id && !live && (
            <Empty>
              Ask anything. Each reply shows which model answered and why, so you can watch the
              local/cloud split as you chat.
            </Empty>
          )}
          {messages.map((m) => (
            <StoredTurn key={m.id} m={m} results={results} />
          ))}
          {live && (
            <>
              <div className="ml-auto max-w-[80%] rounded-xl bg-surface-2 px-3.5 py-2 text-sm whitespace-pre-wrap">
                {live.user}
              </div>
              <StepTimeline steps={live.timeline.steps} live compact={!settings.tools} />
            </>
          )}
          <ErrorNote error={error} />
          <div ref={bottom} />
        </div>

        <div className="border-t border-line p-3">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            <label className="flex items-center gap-1 text-ink-2">
              Strategy
              <Select
                value={settings.strategy}
                onChange={(e) => setSettings({ ...settings, strategy: e.target.value })}
              >
                <option value="">default</option>
                <option value="hybrid">hybrid</option>
                <option value="pre_route">pre-route</option>
                <option value="cascade">cascade</option>
              </Select>
            </label>
            <label className="flex items-center gap-1 text-ink-2">
              Task type
              <Select
                value={settings.task_type}
                onChange={(e) => setSettings({ ...settings, task_type: e.target.value })}
              >
                <option value="">auto</option>
                {TASK_TYPES.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </Select>
            </label>
            <label className="flex items-center gap-1 text-ink-2">
              Force tier
              <Select
                value={settings.force_tier}
                onChange={(e) => setSettings({ ...settings, force_tier: e.target.value })}
              >
                <option value="">no</option>
                <option value="local">local</option>
                <option value="sonnet">sonnet</option>
                <option value="opus">opus</option>
              </Select>
            </label>
            <label className="flex items-center gap-1.5 text-ink-2">
              <input
                type="checkbox"
                checked={settings.tools}
                onChange={(e) => setSettings({ ...settings, tools: e.target.checked })}
              />
              Workspace tools
            </label>
          </div>
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              rows={2}
              placeholder="Message (Enter to send, Shift+Enter for a new line)"
              aria-label="Message"
              className="flex-1 resize-none rounded-md border border-line bg-surface px-3 py-2 text-sm"
            />
            {live ? (
              <Button onClick={() => abort.current?.abort()}>Stop</Button>
            ) : (
              <Button variant="primary" onClick={() => void send()} disabled={!input.trim()}>
                Send
              </Button>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
