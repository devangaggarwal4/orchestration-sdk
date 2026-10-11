// Fetch wrapper with bearer-token auth and a small SSE reader (fetch-based, so
// the Authorization header works; EventSource can't send headers).

const TOKEN_KEY = "hybrid-router-token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: token lives for this page only */
  }
  window.dispatchEvent(new Event("hr-token"));
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  const token = getToken();
  if (token) h.set("Authorization", `Bearer ${token}`);
  return h;
}

async function errorFrom(resp: Response): Promise<ApiError> {
  let detail = resp.statusText;
  try {
    const body = await resp.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* not JSON */
  }
  if (resp.status === 401) window.dispatchEvent(new Event("hr-unauthorized"));
  return new ApiError(resp.status, detail);
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const resp = await fetch(path, {
    ...init,
    headers: headers({ "Content-Type": "application/json", ...(init.headers ?? {}) }),
  });
  if (!resp.ok) throw await errorFrom(resp);
  return (await resp.json()) as T;
}

export const post = <T>(path: string, body: unknown) =>
  api<T>(path, { method: "POST", body: JSON.stringify(body) });

/** Parse an SSE byte stream into JSON events (``data: {...}`` frames). */
export async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (data) yield JSON.parse(data);
    }
  }
}

export async function stream<T>(
  path: string,
  init: RequestInit,
  onEvent: (e: T) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(path, {
    ...init,
    signal,
    headers: headers({ "Content-Type": "application/json", ...(init.headers ?? {}) }),
  });
  if (!resp.ok || !resp.body) throw await errorFrom(resp);
  for await (const event of parseSSE(resp.body)) onEvent(event as T);
}
