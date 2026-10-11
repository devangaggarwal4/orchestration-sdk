import { describe, expect, it } from "vitest";
import { compact, ms, pct, signed, usd, usdTick } from "./format";

describe("format", () => {
  it("formats money, percents and durations", () => {
    expect(usd(0.0123)).toBe("$0.0123");
    expect(usd(12.5)).toBe("$12.50");
    expect(usd(-2)).toBe("−$2.00");
    expect(usd(null)).toBe("–");
    expect(pct(0.256)).toBe("26%");
    expect(ms(850)).toBe("850ms");
    expect(ms(2500)).toBe("2.5s");
    expect(ms(125_000)).toBe("2.1m");
    expect(compact(12_900)).toBe("12.9K");
    expect(signed(-1500, ms)).toBe("−1.5s");
    expect(usdTick(0.0025)).toBe("$0.0025");
    expect(usdTick(0)).toBe("$0");
    expect(usdTick(12)).toBe("$12.00");
  });
});
