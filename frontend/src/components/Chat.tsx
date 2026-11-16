import { useEffect, useRef, useState } from "react";
import { Hand as HandIcon, Mic, Volume2 } from "lucide-react";

export type Message =
  | { id: number; who: "signer"; text: string; signs: string[] }
  | { id: number; who: "speaker"; text: string };


export function groupTurns(messages: Message[]) {
  const turns: { who: Message["who"]; items: Message[] }[] = [];
  for (const m of messages) {
    const last = turns[turns.length - 1];
    if (last && last.who === m.who) last.items.push(m);
    else turns.push({ who: m.who, items: [m] });
  }
  return turns;
}

/** Consecutive messages from the same person, shown under one header like a chat thread. */
export function Turn({ who, items, onSpeak, you }: { who: Message["who"]; items: Message[]; onSpeak: (t: string) => void; you?: Message["who"] }) {
  const signer = who === "signer";
  return (
    <li className="page-enter flex gap-3 border-b border-border pb-4 last:border-0">
      <span
        className={`grid h-11 w-11 shrink-0 place-items-center rounded-full ${
          signer ? "bg-primary-soft text-primary ring-1 ring-primary/40" : "bg-subtle text-fg-2 ring-1 ring-border-strong"
        }`}
        aria-hidden
      >
        {signer ? <HandIcon size={20} /> : <Mic size={20} />}
      </span>
      <div className="min-w-0 flex-1">
        <p className="mb-1.5 text-[14px] font-semibold text-fg">
          {signer ? "Person A" : "Person B"}
          {you === who && <span className="font-normal text-primary"> (you)</span>}{" "}
          <span className="font-normal text-fg-3">– {signer ? "Signs" : "Speaks"}</span>
        </p>
        <ul className="space-y-1.5">
          {items.map((m) => (
            <Line key={m.id} m={m} onSpeak={onSpeak} />
          ))}
        </ul>
      </div>
    </li>
  );
}

function Line({ m, onSpeak }: { m: Message; onSpeak: (t: string) => void }) {
  const signer = m.who === "signer";
  return (
    <li className="page-enter rounded-lg bg-subtle px-3 py-2 ring-1 ring-border">
      <p className="text-[15px] leading-snug text-fg">{m.text}</p>
      <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-fg-3">
        {signer && <span>Signs: {m.signs.join(" · ")}</span>}
        <button className="hover:text-fg" onClick={() => onSpeak(m.text)}>
          <Volume2 size={12} className="mr-1 inline" aria-hidden />
          Speak
        </button>
      </div>
    </li>
  );
}

/** Tap to start recording from the microphone, tap again to stop and transcribe. */
export function useRecorder(onAudio: (b: Blob) => void) {
  const rec = useRef<MediaRecorder | null>(null);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const type = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"].find((t) => MediaRecorder.isTypeSupported(t));
      const r = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      r.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        onAudio(new Blob(chunks, { type: r.mimeType }));
      };
      r.start();
      rec.current = r;
      setRecording(true);
    } catch (e) {
      setError(
        (e as DOMException)?.name === "NotAllowedError"
          ? "Microphone permission was blocked. Allow it from the address bar and try again."
          : "No microphone could be opened.",
      );
    }
  };
  const stop = () => {
    rec.current?.stop();
    rec.current = null;
    setRecording(false);
  };
  useEffect(() => () => rec.current?.stop(), []);
  return { recording, error, start, stop };
}
