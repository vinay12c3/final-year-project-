import { useEffect, useMemo, useState } from "react";
import qrcode from "qrcode-generator";
import { Check, Copy, Globe, Loader2, Users, X } from "lucide-react";
import { api, type Share } from "../lib/api";
import type { Presence } from "../lib/room";
import { Button, Notice } from "./ui";

/** QR code drawn as SVG squares (no HTML injection). */
export function QrCode({ text, size = 200 }: { text: string; size?: number }) {
  const cells = useMemo(() => {
    const q = qrcode(0, "M");
    q.addData(text);
    q.make();
    const n = q.getModuleCount();
    const out: string[] = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (q.isDark(r, c)) out.push(`M${c} ${r}h1v1h-1z`);
    return { n, d: out.join("") };
  }, [text]);
  return (
    <svg width={size} height={size} viewBox={`-2 -2 ${cells.n + 4} ${cells.n + 4}`} role="img" aria-label="QR code for the room link" className="rounded-lg bg-white">
      <path d={cells.d} fill="#0b0d10" shapeRendering="crispEdges" />
    </svg>
  );
}

/**
 * Starts a room and an internet link for it, and shows the link + QR code.
 * Person B opens it on any phone or laptop, anywhere.
 */
export function InviteDialog({
  code,
  presence,
  onClose,
}: {
  code: string;
  presence: Presence;
  onClose: () => void;
}) {
  const [share, setShare] = useState<Share | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let live = true;
    let t = 0;
    const poll = async () => {
      try {
        const s = await api.share();
        if (!live) return;
        setShare(s);
        if (s.state === "off") setShare(await api.shareStart());
        if (s.state !== "on" && s.state !== "failed") t = window.setTimeout(poll, 1000);
      } catch (e) {
        if (live) setShare({ state: "failed", url: null, error: (e as Error).message });
      }
    };
    poll();
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const link = share?.url ? `${share.url}/room/${code}` : null;
  const copy = async () => {
    if (!link) return;
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the link is still visible to copy by hand */
    }
  };
  const retry = async () => setShare(await api.shareStart().catch((e) => ({ state: "failed", url: null, error: (e as Error).message }) as Share));

  return (
    <div role="dialog" aria-modal="true" aria-label="Invite Person B" className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4" onClick={onClose}>
      <div className="page-enter w-full max-w-md rounded-xl bg-surface p-5 shadow-2xl ring-1 ring-border" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-[18px] font-semibold">
              <Globe size={18} className="text-primary" aria-hidden /> Invite Person B
            </h2>
            <p className="mt-1 text-[13px] text-fg-2">
              Room <b className="font-mono tracking-wider text-fg">{code}</b>. Person B scans the code or opens the link on their phone, from anywhere.
            </p>
          </div>
          <button onClick={onClose} aria-label="Close" className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-fg-2 hover:bg-subtle">
            <X size={16} />
          </button>
        </div>

        {(!share || share.state === "starting" || share.state === "downloading" || share.state === "off") && (
          <div className="grid h-[200px] place-items-center rounded-lg bg-subtle text-[13px] text-fg-2">
            <span className="flex items-center gap-2">
              <Loader2 size={16} className="animate-spin text-primary" aria-hidden />
              {share?.state === "downloading" ? "Downloading the sharing tool (one time, about 40 MB)…" : "Creating the internet link…"}
            </span>
          </div>
        )}

        {share?.state === "failed" && (
          <div className="space-y-3">
            <Notice tone="error">{share.error || "Could not create the internet link."} This laptop needs an internet connection.</Notice>
            <Button onClick={retry}>Try again</Button>
          </div>
        )}

        {link && (
          <div className="space-y-4">
            <div className="flex justify-center">
              <QrCode text={link} />
            </div>
            <div className="flex gap-2">
              <input readOnly value={link} onFocus={(e) => e.target.select()} aria-label="Room link"
                className="min-w-0 flex-1 rounded-lg bg-canvas px-3 py-2 font-mono text-[12px] text-fg ring-1 ring-border-strong" />
              <Button variant="primary" onClick={copy}>
                {copied ? <Check size={15} aria-hidden /> : <Copy size={15} aria-hidden />} {copied ? "Copied" : "Copy"}
              </Button>
            </div>
            <p className={`flex items-center gap-2 text-[13px] ${presence.guest ? "text-success" : "text-fg-3"}`}>
              <Users size={15} aria-hidden />
              {presence.guest ? "Person B has joined. You can close this window." : "Waiting for Person B to open the link…"}
            </p>
            <p className="text-[12px] text-fg-3">
              The link works while SignVision is running on this laptop. It only opens this chat; nothing else on the laptop can be seen.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
