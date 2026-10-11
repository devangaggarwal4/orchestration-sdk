import { expect, test } from "@playwright/test";

test("chat shows an escalation, tuning replays a threshold change", async ({ page }) => {
  await page.goto("/chat");

  // An easy message stays local...
  await page.getByLabel("Message").fill("hi, quick question about naming a cat");
  await page.getByRole("button", { name: "Send" }).click();
  const cards = page.getByTestId("route-card");
  await expect(cards.first()).toContainText("Local");

  // ...a hard one escalates, and the card explains why.
  await page.getByLabel("Message").fill("Design the architecture of a distributed lock and its trade-offs");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(cards.nth(1)).toContainText("escalated from");
  await cards.nth(1).getByRole("button", { name: /why/i }).click();
  await expect(cards.nth(1)).toContainText("heuristic");

  // Moving the hard threshold replays recorded routes without model calls.
  await page.getByRole("link", { name: "Tuning" }).click();
  await page.getByLabel("Hard threshold").fill("0.3");
  const whatif = page.getByTestId("whatif");
  await expect(whatif).toContainText("recorded routes would route differently");
  await page.getByRole("button", { name: "Review changes" }).click();
  await expect(page.getByTestId("diff")).toContainText("hard_threshold: 0.3");
});

test("agent run streams steps with tool calls", async ({ page }) => {
  await page.goto("/agent");
  await page.getByLabel("Agent task").fill("Where is escalate_hard_to_opus used?");
  await page.getByRole("button", { name: "Run agent" }).click();
  await expect(page.getByTestId("agent-answer")).toContainText("Here's what I found");
  await expect(page.getByTestId("step").first()).toContainText("search_text");
  await expect(page.getByTestId("run-summary")).toContainText("2 steps");
});

test("dashboard renders after traffic", async ({ page }) => {
  await page.goto("/compare");
  await page.getByLabel("Prompt").fill("Summarize: the cat sat on the mat.");
  await page.getByRole("button", { name: "Compare" }).click();
  await expect(page.getByTestId("leg-all_cloud").getByTestId("route-card")).toBeVisible();
  await page.getByRole("link", { name: "Dashboard" }).click();
  await expect(page.getByTestId("stat-Saved vs all-cloud")).toBeVisible();
  await page.getByRole("button", { name: "Table" }).first().click();
  await expect(page.getByRole("table").first()).toBeVisible();
});
