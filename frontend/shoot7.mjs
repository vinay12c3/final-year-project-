import { chromium } from "playwright-core";
const S = process.argv[2], SP = process.argv[3];
const b = await chromium.launch({ executablePath: process.env.HOME + "/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome",
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${S}/fake.y4m`,
         `--use-file-for-fake-audio-capture=${SP}/speech1.wav%noloop`, "--autoplay-policy=no-user-gesture-required"] });
const ctx = await b.newContext({ viewport: { width: 1440, height: 1000 } });
await ctx.grantPermissions(["camera", "microphone"], { origin: "http://localhost:8000" });
const p = await ctx.newPage();
p.on("pageerror", (e) => console.log("PAGEERROR", e.message));
await p.goto("http://localhost:8000/conversation", { waitUntil: "networkidle" });
await p.waitForTimeout(500);
await p.getByRole("button", { name: "hello", exact: true }).click();
await p.getByRole("button", { name: "how are you", exact: true }).click();
await p.waitForTimeout(500);
console.log("draft:", await p.locator("p.text-\\[24px\\]").innerText());
await p.getByRole("button", { name: "Send", exact: true }).click();
// speaker replies by voice (fake mic plays the test clip)
await p.getByRole("button", { name: /Tap to speak/ }).click();
await p.waitForTimeout(5500);
await p.getByRole("button", { name: /tap to stop/ }).click();
await p.waitForSelector("text=What is your name", { timeout: 30000 });
await p.getByRole("button", { name: "Show as signs" }).last().click();
// signer answers with fingerspelling from the camera (letters mode)
await p.getByRole("radio", { name: "Letters" }).click();
await p.getByRole("button", { name: "name", exact: true }).click();
await p.getByRole("button", { name: "Start camera" }).click();
await p.waitForTimeout(7000);
console.log("draft2:", await p.locator("p.text-\\[24px\\]").innerText());
await p.screenshot({ path: `${S}/h_conv.png` });
await b.close();
