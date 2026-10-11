import { describe, expect, it } from "vitest";
import { parseSSE } from "./client";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      chunks.forEach((x) => c.enqueue(enc.encode(x)));
      c.close();
    },
  });
}

describe("parseSSE", () => {
  it("handles frames split across chunks and skips comments", async () => {
    const out: unknown[] = [];
    for await (const e of parseSSE(streamOf(['data: {"a":', '1}\n\n: ping\n\ndata: {"b":2}\n', "\n"]))) out.push(e);
    expect(out).toEqual([{ a: 1 }, { b: 2 }]);
  });
});
