import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Delete, Eraser, Gauge, Hand as HandIcon, MessageSquareText, Save, ScanSearch, Space, Timer, Volume2 } from "lucide-react";
import { CameraStage } from "../components/CameraStage";
import { Badge, Button, ConfidenceRow, ModeSwitch, Notice, PageHeader, Panel, linkClass, useToast } from "../components/ui";
import { api, wsUrl, type FrameMessage, type Guess, type Hand, type Mode, type Status, type TextMessage } from "../lib/api";
import { speak, useCamera, useFrameSocket } from "../lib/camera";

function usePersistentMode(): [Mode, (m: Mode) => void] {
  const [mode, setMode] = useState<Mode>(() => {
    try {
      return (localStorage.getItem("sv-mode") as Mode) || "letters";
    } catch {
      return "letters";
    }
  });
  const set = (m: Mode) => {
    setMode(m);
    try {
      localStorage.setItem("sv-mode", m);
    } catch {
      /* private mode */
    }
  };
  return [mode, set];
}

const isSign = (g: Guess | null) => !!g && !g.label.startsWith("_");

export default function Translate() {
  const [mode, setMode] = usePersistentMode();
  const cam = useCamera();
  const toast = useToast();
  const [status, setStatus] = useState<Status | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [hands, setHands] = useState<Hand[]>([]);
  const [current, setCurrent] = useState<Guess | null>(null);
  const [top, setTop] = useState<Guess[]>([]);
  const [progress, setProgress] = useState(0);
  const [text, setText] = useState("");
  const [typedKey, setTypedKey] = useState(0);
  const [timing, setTiming] = useState<FrameMessage["timing"] | null>(null);
  const [modelReady, setModelReady] = useState(true);
  const lastText = useRef("");

  useEffect(() => {
    api.status().then(setStatus).catch((e) => setStatusError(e.message));
  }, []);

  const onMessage = useCallback((msg: FrameMessage | TextMessage) => {
    if (msg.type === "text") {
      setText(msg.text);
      lastText.current = msg.text;
      return;
    }
    setHands(msg.hands);
    setModelReady(msg.model_ready);
    setTiming(msg.timing);
    setProgress(msg.progress);
    if (msg.current || !msg.hands.length) {
      setCurrent(msg.current);
      setTop(msg.top);
    }
    if (msg.text !== lastText.current) {
      if (msg.typed && msg.typed !== " ") setTypedKey((k) => k + 1);
      lastText.current = msg.text;
      setText(msg.text);
    }
  }, []);

  // The URL is fixed per camera session: switching mode sends a command instead of
  // reconnecting, so the sentence typed so far is kept.
  const url = useMemo(() => wsUrl(`/ws/translate?mode=${mode}`), [cam.on]); // eslint-disable-line react-hooks/exhaustive-deps
  const sock = useFrameSocket<FrameMessage | TextMessage>(url, cam.videoRef, cam.on, onMessage);

  useEffect(() => {
    sock.send({ type: "mode", mode });
    setCurrent(null);
    setTop([]);
    setProgress(0);
  }, [mode]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!cam.on || (e.target as HTMLElement)?.closest("input,textarea,button")) return;
      if (e.key === " ") {
        e.preventDefault();
        sock.send({ type: "space" });
      } else if (e.key === "Backspace") {
        sock.send({ type: "backspace" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cam.on, sock]);

  const save = async () => {
    if (!text.trim()) return;
    try {
      await api.saveHistory(text, mode, "camera");
      toast.show("Saved to history");
    } catch (e) {
      toast.show((e as Error).message, "error");
    }
  };

  const ms = status?.models[mode];
  const notReady = (status && !ms?.ready) || !modelReady;
  const confident = isSign(current) && (current?.confidence ?? 0) >= 0.7;
  const ranked = top.filter((g) => !g.label.startsWith("_")).slice(0, 3);
  const bar = confident || mode === "words" ? progress : 0;

  const lastToken = mode === "letters" ? text.slice(-1) : (text.trimEnd().split(" ").pop() ?? "");
  const head = text.slice(0, text.trimEnd().length - lastToken.length);
  const tail = text.slice(text.trimEnd().length);

  return (
    <div>
      <PageHeader
        title="Live translation"
        description={
          <>
            {mode === "letters"
              ? "Hold each sign still until the bar fills; lower your hands to add a space."
              : "Sign one word at a time with a short pause between words."}{" "}
            <Link to="/guide" className={linkClass}>View sign guide</Link>
          </>
        }
        actions={<ModeSwitch mode={mode} onChange={setMode} />}
      />

      {statusError && <div className="mb-4"><Notice tone="error">{statusError}</Notice></div>}

      <div className="stagger grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <Panel bodyClassName="">
          <CameraStage
            videoRef={cam.videoRef}
            on={cam.on}
            error={cam.error}
            hands={hands}
            onStart={cam.start}
            onStop={cam.stop}
            connected={sock.connected}
          />
          <dl className="grid grid-cols-3 divide-x divide-border border-t border-border">
            {([
              ["Frame rate", cam.on ? `${sock.fps} fps` : "–", Gauge, "bg-primary-soft text-primary"],
              ["Latency", timing ? `${Math.round(timing.total_ms)} ms` : "–", Timer, "bg-primary-soft text-primary"],
              ["Hands", cam.on ? String(hands.length) : "–", HandIcon, "bg-primary-soft text-primary"],
            ] as const).map(([k, v, Icon, chip]) => (
              <div key={k} className="flex items-center gap-3 px-4 py-2.5">
                <span className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg ${chip}`}>
                  <Icon size={16} aria-hidden />
                </span>
                <div>
                  <dt className="text-[12px] text-fg-3">{k}</dt>
                  <dd className="text-[14px] font-semibold text-fg">{v}</dd>
                </div>
              </div>
            ))}
          </dl>
        </Panel>

        <Panel
          title="Detected sign"
          icon={ScanSearch}
          accent="violet"
          actions={ms?.ready ? <Badge tone="success">Model ready</Badge> : null}
        >
          {notReady ? (
            <div className="space-y-3 py-2">
              <p className="font-semibold">The {mode} model isn't trained yet</p>
              <p className="text-fg-2">Record a few examples of each sign, then train the model.</p>
              <Link to="/record" className={linkClass}>Go to Record and train</Link>
            </div>
          ) : (
            <div className="space-y-5">
              <div
                className={`relative flex min-h-[112px] items-center justify-between gap-3 overflow-hidden rounded-xl px-4 transition-colors duration-300 ${
                  confident ? "bg-primary-soft text-fg ring-1 ring-inset ring-primary/60" : "bg-subtle ring-1 ring-inset ring-border"
                }`}
              >
                <span
                  key={current?.label ?? "none"}
                  aria-live="polite"
                  className={`pop leading-none font-bold break-words ${
                    (current?.display.length ?? 0) > 3 ? "text-[26px]" : "text-[64px]"
                  } ${confident ? "" : "text-fg-3"}`}
                >
                  {isSign(current) ? current!.display : "—"}
                </span>
                {isSign(current) && (
                  <span className={`rounded-full px-2.5 py-1 text-[13px] font-semibold ${confident ? "bg-primary text-on-primary" : "text-fg-3"}`}>
                    {Math.round(current!.confidence * 100)}%
                  </span>
                )}
              </div>

              <div>
                <div className="mb-1.5 flex justify-between text-[12px] text-fg-3">
                  <span>{mode === "letters" ? "Hold to type" : "Signing window"}</span>
                  <span>{Math.round(bar * 100)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-subtle">
                  <div
                    className={`bar-fill h-full rounded-full transition-[width] duration-100 ${bar >= 1 ? "bar-full" : ""}`}
                    style={{ width: `${bar * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <p className="mb-1 text-[12px] font-medium text-fg-3">Top predictions</p>
                {ranked.length ? (
                  ranked.map((g, i) => <ConfidenceRow key={g.label} label={g.display} value={g.confidence} strong={i === 0} />)
                ) : (
                  <p className="py-1.5 text-[13px] text-fg-3">
                    {cam.on ? (hands.length ? "Reading the sign…" : "Show your hands to the camera.") : "Start the camera to begin."}
                  </p>
                )}
              </div>
            </div>
          )}
        </Panel>
      </div>

      <Panel
        className="page-enter mt-4"
        title="Transcript"
        icon={MessageSquareText}
        accent="pink"
        actions={
          <>
            <Button size="sm" onClick={() => sock.send({ type: "space" })} disabled={!cam.on} title="Space bar">
              <Space size={15} aria-hidden /> <span className="sr-only sm:not-sr-only">Space</span>
            </Button>
            <Button size="sm" onClick={() => sock.send({ type: "backspace" })} disabled={!cam.on || !text} title="Backspace">
              <Delete size={15} aria-hidden /> <span className="sr-only sm:not-sr-only">Delete</span>
            </Button>
            <Button size="sm" onClick={() => sock.send({ type: "clear" })} disabled={!cam.on || !text}>
              <Eraser size={15} aria-hidden /> <span className="sr-only sm:not-sr-only">Clear</span>
            </Button>
            <Button size="sm" onClick={save} disabled={!text.trim()}>
              <Save size={15} aria-hidden /> <span className="sr-only sm:not-sr-only">Save</span>
            </Button>
            <Button size="sm" variant="primary" onClick={() => speak(text) || toast.show("This browser has no speech voices.", "error")} disabled={!text.trim()}>
              <Volume2 size={15} aria-hidden /> Speak
            </Button>
          </>
        }
        bodyClassName="px-4 py-5"
      >
        <p className="min-h-[44px] text-[28px] leading-snug font-semibold break-words sm:text-[34px]" aria-live="polite">
          {text ? (
            <>
              {head}
              <span key={typedKey} className="typed">{lastToken}</span>
              {tail}
            </>
          ) : (
            <span className="text-fg-3/60">Recognised {mode === "letters" ? "letters" : "words"} appear here</span>
          )}
          <span className="caret bg-primary ml-0.5 inline-block h-[0.9em] w-[3px] translate-y-[0.12em] rounded-full" aria-hidden />
        </p>
      </Panel>
      {toast.node}
    </div>
  );
}
