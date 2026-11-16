import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Loader2, MessagesSquare, Mic, Send, Square, Volume2, VolumeX } from "lucide-react";
import { groupTurns, Turn, useRecorder, type Message } from "../components/Chat";
import { Logo } from "../components/HandMark";
import { Button, Notice, useToast } from "../components/ui";
import { api } from "../lib/api";
import { speak } from "../lib/camera";
import { useRoom } from "../lib/room";

/** Person B's view of a conversation room, opened from the invite link on their own phone. */
export default function Room() {
  const code = (useParams().code ?? "").toUpperCase();
  const toast = useToast();
  const [joined, setJoined] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [reply, setReply] = useState("");
  const [busy, setBusy] = useState(false);
  const [voice, setVoice] = useState(true);
  const voiceRef = useRef(true);
  voiceRef.current = voice;
  const nextId = useRef(1);
  const end = useRef<HTMLDivElement | null>(null);

  const room = useRoom(joined ? code : null, "guest", {
    onHistory: (ms) => setMessages(ms.map((m) => ({ ...m, id: nextId.current++, signs: m.signs ?? [] }) as Message)),
    onMessage: (m) => {
      setMessages((ms) => [...ms, { ...m, id: nextId.current++, signs: m.signs ?? [] } as Message]);
      if (m.who === "signer" && voiceRef.current) speak(m.text);    // Person A's signs, read aloud
    },
  });

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  const send = (text: string) => {
    const t = text.trim();
    if (!t) return;
    setMessages((ms) => [...ms, { id: nextId.current++, who: "speaker", text: t }]);
    room.send("speaker", t);
  };

  const mic = useRecorder(async (blob) => {
    setBusy(true);
    try {
      const r = await api.roomTranscribe(code, blob);
      if (r.text) send(r.text);
      else toast.show("No speech was heard. Speak a little louder and try again.", "error");
    } catch (e) {
      toast.show((e as Error).message, "error");
    } finally {
      setBusy(false);
    }
  });

  if (!joined) {
    return (
      <div className="grid min-h-[100dvh] place-items-center bg-canvas p-6">
        <div className="page-enter w-full max-w-sm text-center">
          <div className="mb-6 flex justify-center"><Logo dark /></div>
          <MessagesSquare size={40} className="float mx-auto mb-4 text-primary" aria-hidden />
          <h1 className="text-[22px] font-semibold">You're invited to a conversation</h1>
          <p className="mt-2 text-[14px] text-fg-2">
            Person A signs and their sentences appear here and are read aloud. You reply by speaking or typing.
          </p>
          <p className="mt-3 font-mono text-[13px] tracking-widest text-fg-3">ROOM {code}</p>
          {/* the tap also lets the browser play speech later */}
          <Button variant="primary" className="mt-6 w-full justify-center py-3 text-[16px]" onClick={() => { speak(" "); setJoined(true); }}>
            Join conversation
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-[100dvh] flex-col bg-canvas">
      <header className="flex items-center justify-between gap-3 border-b border-border bg-sidebar px-4 py-3">
        <Logo dark />
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-[12px] text-fg-2">
            <span className={`h-2 w-2 rounded-full ${room.connected ? (room.presence.host ? "ping bg-success" : "bg-warning") : "bg-danger"}`} aria-hidden />
            {!room.connected ? "Connecting…" : room.presence.host ? "Person A connected" : "Person A is away"}
          </span>
          <button onClick={() => setVoice((v) => !v)} aria-label={voice ? "Stop reading messages aloud" : "Read messages aloud"} aria-pressed={voice}
            className="grid h-9 w-9 place-items-center rounded-lg text-fg-2 ring-1 ring-border-strong hover:text-fg">
            {voice ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>
        </div>
      </header>

      <main className="mx-auto w-full max-w-2xl flex-1 overflow-y-auto px-4 py-4">
        {room.error && <Notice tone="error">{room.error}</Notice>}
        {!room.error && messages.length === 0 && (
          <p className="py-16 text-center text-[14px] text-fg-3">Waiting for Person A to sign…</p>
        )}
        <ul className="space-y-4" aria-live="polite">
          {groupTurns(messages).map((t) => (
            <Turn key={t.items[0].id} who={t.who} items={t.items} onSpeak={(x) => speak(x)} you="speaker" />
          ))}
        </ul>
        <div ref={end} />
      </main>

      <footer className="border-t border-border bg-surface px-4 pb-[max(12px,env(safe-area-inset-bottom))] pt-3">
        <div className="mx-auto max-w-2xl space-y-2">
          {mic.error && <Notice tone="error">{mic.error}</Notice>}
          <Button variant="primary" className="w-full justify-center py-3 text-[15px]" disabled={busy || !!room.error}
            onClick={mic.recording ? mic.stop : mic.start}>
            {busy ? <><Loader2 size={17} className="animate-spin" aria-hidden /> Writing what you said…</>
              : mic.recording ? <><Square size={15} aria-hidden /> Tap to stop and send</>
              : <><Mic size={17} aria-hidden /> Tap to speak a reply</>}
          </Button>
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); send(reply); setReply(""); }}>
            <input value={reply} onChange={(e) => setReply(e.target.value)} placeholder="Or type a reply…" aria-label="Type a reply"
              className="min-w-0 flex-1 rounded-lg bg-canvas px-3 py-2.5 text-[15px] text-fg ring-1 ring-border-strong placeholder:text-fg-3 focus:ring-primary" />
            <Button type="submit" variant="primary" disabled={!reply.trim() || !!room.error} aria-label="Send reply">
              <Send size={15} aria-hidden /> Send
            </Button>
          </form>
        </div>
      </footer>
      {toast.node}
    </div>
  );
}
