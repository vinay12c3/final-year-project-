import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Globe, Hand as HandIcon, Loader2, MessagesSquare, Mic, RotateCcw, Save, Send, Square, Undo2, Volume2, X } from "lucide-react";
import { CameraStage } from "../components/CameraStage";
import { groupTurns, Turn, useRecorder, type Message } from "../components/Chat";
import { InviteDialog } from "../components/Invite";
import { useRoom } from "../lib/room";
import { Badge, Button, Notice, PageHeader, Panel, Segmented, linkClass, useToast } from "../components/ui";
import { api, wsUrl, type FrameMessage, type Guess, type Hand, type Phrases, type SignToken, type Status, type TextMessage } from "../lib/api";
import { speak, useCamera, useFrameSocket } from "../lib/camera";

type NewMessage = { who: "signer"; text: string; signs: string[] } | { who: "speaker"; text: string };
const AUTO_SEND_MS = 2500;
const pretty = (label: string) => label.replace(/_/g, " ");
const tokenText = (t: SignToken) => (t.kind === "word" ? pretty(t.label) : t.text);
type ConvMode = "sentences" | "words";

export default function Conversation() {
  const cam = useCamera();
  const toast = useToast();
  const [mode, setMode] = useState<ConvMode>("sentences");
  const [unsure, setUnsure] = useState<Guess[]>([]);
  const [signing, setSigning] = useState(0);
  const [finishing, setFinishing] = useState(0);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const [phrases, setPhrases] = useState<Phrases>({});
  const [hands, setHands] = useState<Hand[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [tokens, setTokens] = useState<SignToken[]>([]);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [reply, setReply] = useState("");
  const [busy, setBusy] = useState(false);
  const [autoSend, setAutoSend] = useState(true);
  const [speakAloud, setSpeakAloud] = useState(true);
  const [pauseMs, setPauseMs] = useState(0);
  const spellOpen = useRef(false);
  const autoRef = useRef(true);
  const speakRef = useRef(true);
  const lastTokenAt = useRef(0);
  autoRef.current = autoSend;
  speakRef.current = speakAloud;
  const lastHandAt = useRef(performance.now());
  const chatEnd = useRef<HTMLLIElement | null>(null);
  const nextId = useRef(1);
  const [roomCode, setRoomCode] = useState<string | null>(null);
  const [inviteOpen, setInviteOpen] = useState(false);
  const room = useRoom(roomCode, "host", {
    // Person B's replies from their own phone
    onMessage: (m) => setMessages((ms) => [...ms, { ...m, id: nextId.current++, signs: m.signs ?? [] } as Message]),
  });
  const roomSend = useRef(room.send);
  roomSend.current = roomCode ? room.send : () => undefined;
  /** Adds a message to the chat here, and to Person B's screen when a room is open. */
  const post = useCallback((m: NewMessage) => {
    setMessages((ms) => [...ms, { ...m, id: nextId.current++ } as Message]);
    roomSend.current(m.who, m.text, m.who === "signer" ? m.signs : undefined);
  }, []);
  const invite = async () => {
    try {
      if (!roomCode) setRoomCode((await api.createRoom()).code);
      setInviteOpen(true);
    } catch (e) {
      toast.show((e as Error).message, "error");
    }
  };

  useEffect(() => {
    api.status().then((s) => {
      setStatus(s);
      if (!s.models.sentences?.ready && s.models.words.ready) setMode("words");
    }).catch(() => undefined);
    api.phrases().then(setPhrases).catch(() => undefined);
  }, []);

  useEffect(() => {
    lastTokenAt.current = performance.now();
  }, [tokens.length]);

  useEffect(() => {
    let live = true;
    // the server turns the signs into a sentence; if it can't (e.g. an older server), join them as they are
    const fallback = tokens.map(tokenText).join(" ");
    if (!tokens.length) setDraft("");
    else {
      setDraft((d) => d || fallback);
      api.compose(tokens).then((r) => live && setDraft(r.sentence || fallback)).catch(() => live && setDraft(fallback));
    }
    return () => {
      live = false;
    };
  }, [tokens]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const addToken = useCallback((t: SignToken) => {
    spellOpen.current = false;
    setTokens((ts) => [...ts, t]);
  }, []);

  const onMessage = useCallback((msg: FrameMessage | TextMessage) => {
    if (msg.type !== "frame") return;
    setHands(msg.hands);
    if (msg.hands.length) lastHandAt.current = performance.now();
    if (msg.mode === "sentences") {
      setSigning(msg.signing ?? 0);
      setFinishing(msg.hands.length ? 0 : msg.progress);
      if (msg.signing && msg.top.length) setCurrent(msg.top[0].display);
      else if (!msg.hands.length && !msg.progress) setCurrent(null);
      if (msg.typed && msg.typed_text) {
        const text = msg.typed_text;
        setUnsure([]);
        if (autoRef.current) {
          // a recognised sentence goes straight into the chat
          post({ who: "signer", text, signs: [text] });
          if (speakRef.current) speak(text);
        } else {
          setTokens((ts) => [...ts, { kind: "sentence", label: msg.typed!, text }]);
        }
      } else if (msg.current && !msg.typed) {
        setUnsure(msg.top);                         // finished signing but not sure which sentence
      }
      return;
    }
    if (msg.current || !msg.hands.length) setCurrent(msg.current && !msg.current.label.startsWith("_") ? msg.current.display : null);
    const typed = msg.typed;
    if (!typed) return;
    if (msg.mode === "words") {
      if (typed !== " " && !typed.startsWith("_")) {
        spellOpen.current = false;
        setTokens((ts) => [...ts, { kind: "word", label: typed }]);
      }
    } else if (typed === " ") {
      spellOpen.current = false;
    } else {
      setTokens((ts) => {
        const last = ts[ts.length - 1];
        if (spellOpen.current && last?.kind === "spell") return [...ts.slice(0, -1), { kind: "spell", text: last.text + typed }];
        spellOpen.current = true;
        return [...ts, { kind: "spell", text: typed }];
      });
    }
  }, []);

  const url = useMemo(() => wsUrl(`/ws/translate?mode=${mode}`), [cam.on]); // eslint-disable-line react-hooks/exhaustive-deps
  const sock = useFrameSocket<FrameMessage | TextMessage>(url, cam.videoRef, cam.on, onMessage);
  useEffect(() => {
    sock.send({ type: "mode", mode });
    setCurrent(null);
    setUnsure([]);
  }, [mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const say = useCallback((t: string) => speak(t) || toast.show("This browser has no speech voices.", "error"), [toast]);

  const sendSigned = useCallback(() => {
    if (!draft) return;
    post({ who: "signer", text: draft, signs: tokens.map(tokenText) });
    if (speakAloud) speak(draft);
    setTokens([]);
    spellOpen.current = false;
    sock.send({ type: "clear" });
  }, [draft, tokens, speakAloud, sock]);

  // Auto-send: when the signer lowers their hands for a few seconds, the sentence is sent.
  useEffect(() => {
    if (!autoSend || !tokens.length) {
      setPauseMs(0);
      return;
    }
    const t = window.setInterval(() => {
      // sent automatically a moment after the last recognised sign (or tapped sentence)
      const p = performance.now() - lastTokenAt.current;
      setPauseMs(p);
      if (p >= (mode === "sentences" ? 800 : AUTO_SEND_MS) && draft) sendSigned();
    }, 200);
    return () => clearInterval(t);
  }, [autoSend, tokens.length, draft, sendSigned, mode]);

  const addSpeaker = (text: string) => {
    const t = text.trim();
    if (t) post({ who: "speaker", text: t });
  };

  const mic = useRecorder(async (blob) => {
    setBusy(true);
    try {
      const r = await api.transcribe(blob);
      if (r.text) addSpeaker(r.text);
      else toast.show("No speech was heard. Speak a little louder and try again.", "error");
    } catch (e) {
      toast.show((e as Error).message, "error");
    } finally {
      setBusy(false);
    }
  });

  const save = async () => {
    const text = messages.map((m) => `${m.who === "signer" ? "Person A (signs)" : "Person B (speaks)"}: ${m.text}`).join("\n");
    try {
      await api.saveHistory(text, "conversation", "conversation");
      toast.show("Conversation saved to history");
    } catch (e) {
      toast.show((e as Error).message, "error");
    }
  };

  const ms = mode === "sentences" ? status?.models.sentences : status?.models.words;
  const known = (status?.models.words.labels ?? []).filter((l) => !l.startsWith("_"));
  const suggestions = known.length ? known : Object.keys(phrases);
  const sentenceTexts = status?.models.sentences?.texts ?? {};
  const sentenceList = useMemo(() => {
    const q = search.trim().toLowerCase();
    const all = Object.entries(sentenceTexts).sort((a, b) => a[1].localeCompare(b[1]));
    return q ? all.filter(([, t]) => t.toLowerCase().includes(q)) : all;
  }, [sentenceTexts, search]);

  return (
    <div>
      <PageHeader
        title="Conversation"
        description={
          <>
            The signer signs and the app writes a full sentence and speaks it. The other person replies by voice or by typing.{" "}
            <Link to="/guide" className={linkClass}>View sign guide</Link>
          </>
        }
        actions={
          <>
            <Segmented<ConvMode>
              label="What Person A signs"
              value={mode}
              onChange={setMode}
              options={[
                { value: "sentences", label: "Sentences", hint: "One sign = one whole sentence" },
                { value: "words", label: "Words", hint: "Greeting words joined into a sentence" },
              ]}
            />
            <Button onClick={invite} variant={roomCode ? "secondary" : "primary"}>
              <Globe size={15} aria-hidden /> {roomCode ? `Room ${roomCode}` : "Invite Person B"}
            </Button>
            <Button onClick={() => setMessages([])} disabled={!messages.length}>
              <RotateCcw size={15} aria-hidden /> New
            </Button>
            <Button onClick={save} disabled={!messages.length}>
              <Save size={15} aria-hidden /> Save
            </Button>
          </>
        }
      />

      {status && !ms?.ready && (
        <div className="mb-4">
          <Notice tone="warning">
            The {mode} model isn't trained yet. <Link to="/record" className={linkClass}>Record and train</Link> it first.
          </Notice>
        </div>
      )}

      <div className="stagger grid gap-4 lg:grid-cols-[minmax(0,1fr)_420px]">
        {/* ---------------------------------------------------------- signer */}
        <div className="space-y-4">
          <Panel bodyClassName="">
            <CameraStage
              videoRef={cam.videoRef}
              on={cam.on}
              error={cam.error}
              hands={hands}
              onStart={cam.start}
              onStop={cam.stop}
              connected={sock.connected}
              overlay={
                mode === "sentences" && (signing > 0 || finishing > 0) ? (
                  <div className="absolute inset-x-3 bottom-3 rounded-lg bg-black/70 px-3 py-2 text-white backdrop-blur">
                    <p className="flex items-center justify-between text-[12px] text-white/70">
                      <span>{finishing > 0 ? "Hands lowered – reading the sentence…" : `Signing… ${signing.toFixed(1)} s`}</span>
                      {finishing > 0 && (
                        <span className="h-1.5 w-24 overflow-hidden rounded-full bg-white/15">
                          <span className="bar-fill block h-full" style={{ width: `${finishing * 100}%` }} />
                        </span>
                      )}
                    </p>
                    {current && <p className="mt-0.5 truncate text-[16px] font-semibold">{current}</p>}
                  </div>
                ) : current ? (
                  <span key={current} className="pop absolute bottom-3 left-3 rounded-lg bg-black/70 px-3 py-1.5 text-[18px] font-semibold text-white">
                    {current}
                  </span>
                ) : null
              }
            />
          </Panel>

          <Panel
            title="Person A is signing"
            icon={HandIcon}
            actions={
              <>
                <Button size="sm" onClick={() => setTokens((t) => t.slice(0, -1))} disabled={!tokens.length}>
                  <Undo2 size={15} aria-hidden /> Undo
                </Button>
                <Button size="sm" onClick={() => setTokens([])} disabled={!tokens.length}>
                  <X size={15} aria-hidden /> Clear
                </Button>
                <Button size="sm" variant="primary" onClick={sendSigned} disabled={!draft}>
                  <Send size={15} aria-hidden /> Send
                </Button>
              </>
            }
          >
            <div className="flex min-h-[28px] flex-wrap gap-1.5">
              {tokens.length ? (
                tokens.map((t, i) => (
                  <span key={i} className="page-enter">
                    <Badge tone={t.kind === "spell" ? "neutral" : "primary"}>{tokenText(t)}</Badge>
                  </span>
                ))
              ) : (
                <span className="text-[13px] text-fg-3">
                  {mode === "sentences"
                    ? cam.on
                      ? "Raise your hands, sign the whole sentence, then lower your hands."
                      : "Start the camera and sign a sentence, or pick one below."
                    : cam.on
                      ? "Sign a word, then lower your hands briefly before the next one."
                      : "Start the camera and sign, or tap a sign below."}
                </span>
              )}
            </div>
            <p key={draft} className="page-enter mt-3 min-h-[36px] text-[24px] leading-snug font-semibold text-fg">
              {draft || <span className="text-fg-3/60">The sentence appears here</span>}
            </p>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-[12px] text-fg-3">
              <div className="flex flex-wrap gap-4">
                <label className="flex items-center gap-1.5">
                  <input type="checkbox" checked={autoSend} onChange={(e) => setAutoSend(e.target.checked)} className="accent-primary" />
                  {mode === "sentences" ? "Send each sentence to the chat automatically" : "Send automatically 2.5 s after the last word"}
                </label>
                <label className="flex items-center gap-1.5">
                  <input type="checkbox" checked={speakAloud} onChange={(e) => setSpeakAloud(e.target.checked)} className="accent-primary" />
                  Speak aloud
                </label>
              </div>
              {autoSend && tokens.length > 0 && pauseMs > 400 && (
                <span className="flex items-center gap-2 text-primary">
                  Sending
                  <span className="h-1.5 w-20 overflow-hidden rounded-full bg-subtle">
                    <span className="bar-fill block h-full" style={{ width: `${Math.min(100, (pauseMs / AUTO_SEND_MS) * 100)}%` }} />
                  </span>
                </span>
              )}
            </div>

            {mode === "sentences" && unsure.length > 0 && (
              <div className="page-enter mt-3 rounded-lg bg-warning-soft px-3 py-2.5 ring-1 ring-warning/30">
                <p className="mb-2 text-[12px] text-fg-2">Not sure which sentence that was. Did you mean:</p>
                <div className="flex flex-wrap gap-1.5">
                  {unsure.map((g) => (
                    <button
                      key={g.label}
                      onClick={() => {
                        addToken({ kind: "sentence", label: g.label, text: g.display });
                        setUnsure([]);
                      }}
                      className="press rounded-md bg-surface px-2 py-1 text-[13px] text-fg ring-1 ring-border hover:ring-primary"
                    >
                      {g.display} <span className="text-fg-3">{Math.round(g.confidence * 100)}%</span>
                    </button>
                  ))}
                  <button onClick={() => setUnsure([])} className="px-2 text-[12px] text-fg-3 hover:text-fg">
                    None of these
                  </button>
                </div>
              </div>
            )}

            {mode === "sentences" ? (
              <div className="mt-4 border-t border-border pt-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <p className="text-[12px] text-fg-3">
                    {Object.keys(sentenceTexts).length} sentences the camera knows. Tap one if the camera missed it.{" "}
                    <Link to="/guide" className={linkClass}>How to sign them</Link>
                  </p>
                  <input
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search sentences…"
                    aria-label="Search sentences"
                    className="h-8 w-48 rounded-lg bg-subtle px-3 text-[13px] text-fg ring-1 ring-border outline-none placeholder:text-fg-3 focus:ring-primary"
                  />
                </div>
                <div className="flex max-h-[148px] flex-wrap gap-1.5 overflow-y-auto pr-1">
                  {sentenceList.map(([label, text]) => (
                    <button
                      key={label}
                      onClick={() => addToken({ kind: "sentence", label, text })}
                      className="press rounded-md bg-subtle px-2 py-1 text-[12px] text-fg-2 ring-1 ring-border transition-colors hover:text-primary hover:ring-primary/50"
                    >
                      {text}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
            <div className="mt-4 border-t border-border pt-3">
              <p className="mb-2 text-[12px] text-fg-3">Tap a sign to add it if the camera missed it</p>
              <div className="flex flex-wrap gap-1.5">
                {suggestions.map((l) => (
                  <button
                    key={l}
                    onClick={() => addToken({ kind: "word", label: l })}
                    className="press rounded-md bg-subtle px-2 py-1 text-[12px] text-fg-2 ring-1 ring-border transition-colors hover:text-primary hover:ring-primary/50"
                    title={phrases[l]?.text}
                  >
                    {pretty(l)}
                  </button>
                ))}
              </div>
            </div>
            )}
          </Panel>
        </div>

        {/* ---------------------------------------------------------- chat */}
        <Panel
          title="Live conversation"
          icon={MessagesSquare}
          actions={
            roomCode && (
              <button onClick={() => setInviteOpen(true)} className="flex items-center gap-1.5 text-[12px] text-fg-2 hover:text-fg">
                <span className={`h-2 w-2 rounded-full ${room.presence.guest ? "ping bg-success" : room.connected ? "bg-warning" : "bg-danger"}`} aria-hidden />
                {room.presence.guest ? "Person B connected" : room.connected ? "Waiting for Person B" : "Reconnecting…"}
              </button>
            )
          }
          bodyClassName="flex h-[640px] flex-col lg:h-[calc(100vh-190px)] lg:min-h-[560px]">
          <ul className="flex-1 space-y-4 overflow-y-auto p-4" aria-live="polite">
            {messages.length === 0 && (
              <li className="py-10 text-center text-[13px] text-fg-3">
                <MessagesSquare size={28} className="float mx-auto mb-3 text-primary/70" aria-hidden />
                Messages from both people appear here.
                <br />
                Try signing <b className="text-fg-2">{mode === "sentences" ? "Hi, how are you?" : "hello"}</b>
                {mode === "words" && <> and then <b className="text-fg-2">how are you</b></>}.
              </li>
            )}
            {groupTurns(messages).map((t) => (
              <Turn key={t.items[0].id} who={t.who} items={t.items} onSpeak={say} />
            ))}
            <li ref={chatEnd} aria-hidden />
          </ul>

          <div className="space-y-2 border-t border-border p-3">
            {mic.error && <Notice tone="error">{mic.error}</Notice>}
            {status?.speech && !status.speech.downloaded && (
              <p className="text-[12px] text-warning">The speech model downloads on first use (about 145 MB, needs internet once).</p>
            )}
            <button
              onClick={mic.recording ? mic.stop : mic.start}
              disabled={busy}
              className={`press flex h-12 w-full items-center justify-center gap-2 rounded-lg text-[14px] font-semibold transition-colors disabled:opacity-60 ${
                mic.recording ? "bg-danger text-white" : "bg-primary text-on-primary hover:bg-primary-hover"
              }`}
            >
              {busy ? (
                <>
                  <Loader2 size={17} className="animate-spin" aria-hidden /> Converting speech to text…
                </>
              ) : mic.recording ? (
                <>
                  <span className="ping h-2.5 w-2.5 rounded-full bg-white" aria-hidden />
                  <Square size={15} aria-hidden /> Listening… tap to stop
                </>
              ) : (
                <>
                  <Mic size={17} aria-hidden /> Tap to speak a reply
                </>
              )}
            </button>
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                addSpeaker(reply);
                setReply("");
              }}
            >
              <input
                value={reply}
                onChange={(e) => setReply(e.target.value)}
                placeholder="Or type a reply…"
                className="h-9 min-w-0 flex-1 rounded-lg bg-subtle px-3 text-[14px] text-fg ring-1 ring-border outline-none placeholder:text-fg-3 focus:ring-primary"
              />
              <Button type="submit" variant="primary" disabled={!reply.trim()} aria-label="Send reply">
                <Send size={15} aria-hidden /> Send
              </Button>
            </form>
          </div>
        </Panel>
      </div>
      {toast.node}
      {inviteOpen && roomCode && <InviteDialog code={roomCode} presence={room.presence} onClose={() => setInviteOpen(false)} />}
    </div>
  );
}
