import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Dumbbell, ListChecks, Plus, Undo2 } from "lucide-react";
import { CameraStage } from "../components/CameraStage";
import { Badge, Button, Notice, PageHeader, Panel, Segmented, useToast } from "../components/ui";
import { api, kindOf, wsUrl, type Hand, type RecMode, type TrainJob, type Vocabulary } from "../lib/api";
import { useCamera, useFrameSocket } from "../lib/camera";

type RecordMessage =
  | { type: "frame"; hands: Hand[]; recording: boolean; elapsed?: number; captured?: number }
  | { type: "saved"; hands: Hand[]; label: string; frames: number; recorded: Record<string, number> }
  | { type: "rejected"; hands: Hand[]; reason: string };

const SECONDS: Record<RecMode, number> = { letters: 3, words: 2.5, sentences: 6 };
const GOAL: Record<RecMode, number> = { letters: 5, words: 15, sentences: 5 };
const MODE_NAME: Record<RecMode, string> = { letters: "letters", words: "words", sentences: "sentences" };

const plain = (l: string) => (l === "_idle" ? "No sign (idle)" : l.replace(/_/g, " "));

function readName() {
  try {
    return localStorage.getItem("sv-signer") ?? "";
  } catch {
    return "";
  }
}

const inputClass =
  "h-9 rounded-lg bg-surface px-3 text-[14px] ring-1 ring-inset ring-border-strong outline-none placeholder:text-fg-3 focus:ring-2 focus:ring-primary";

export default function Record() {
  const [mode, setMode] = useState<RecMode>("sentences");
  const [vocab, setVocab] = useState<Vocabulary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [custom, setCustom] = useState("");
  const [signer, setSigner] = useState(readName);
  const [hands, setHands] = useState<Hand[]>([]);
  const [phase, setPhase] = useState<"idle" | "countdown" | "recording">("idle");
  const [count, setCount] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [job, setJob] = useState<TrainJob | null>(null);
  const cam = useCamera();
  const toast = useToast();
  const logRef = useRef<HTMLPreElement | null>(null);
  const kind = kindOf(mode);
  const texts = vocab?.sentences?.texts ?? {};
  const display = (l: string) => (mode === "sentences" ? texts[l] ?? l : plain(l));

  const loadVocab = useCallback(() => api.vocabulary().then(setVocab).catch((e) => setError(e.message)), []);
  useEffect(() => {
    loadVocab();
  }, [loadVocab]);

  const labels = useMemo(() => {
    if (!vocab) return [];
    const v = vocab[mode];
    if (!v) return [];
    const all = [...new Set([...v.suggested, ...Object.keys(v.recorded)])];
    return mode === "sentences" ? all.sort((a, b) => (texts[a] ?? a).localeCompare(texts[b] ?? b)) : all;
  }, [vocab, mode]);
  const recorded = vocab?.[mode]?.recorded ?? {};
  const publicCounts = vocab?.[mode]?.public ?? {};

  useEffect(() => {
    setLabel(null);
    setPhase("idle");
  }, [mode]);

  const onMessage = useCallback(
    (msg: RecordMessage) => {
      setHands(msg.hands);
      if (msg.type === "frame") {
        if (msg.recording) setElapsed(msg.elapsed ?? 0);
      } else if (msg.type === "saved") {
        setPhase("idle");
        setVocab((v) => (v ? { ...v, [mode]: { ...v[mode], recorded: msg.recorded } } : v));
        toast.show(`Saved recording ${msg.recorded[msg.label] ?? ""} (${msg.frames} frames)`);
        if (mode === "sentences") {
          setLabel(msg.label);                     // a new sentence now has its own label
          loadVocab();
        }
      } else if (msg.type === "rejected") {
        setPhase("idle");
        toast.show(msg.reason, "error");
      }
    },
    [mode, toast, loadVocab],
  );
  const sock = useFrameSocket<RecordMessage>(wsUrl("/ws/record"), cam.videoRef, cam.on, onMessage, 15);

  const record = () => {
    if (!label || phase !== "idle") return;
    try {
      localStorage.setItem("sv-signer", signer);
    } catch {
      /* ignore */
    }
    setPhase("countdown");
    let n = 3;
    setCount(n);
    const t = setInterval(() => {
      n -= 1;
      if (n > 0) setCount(n);
      else {
        clearInterval(t);
        setElapsed(0);
        setPhase("recording");
        sock.send({ type: "start", kind, label, seconds: SECONDS[mode], who: signer });
      }
    }, 800);
  };

  const undo = async () => {
    if (!label) return;
    try {
      const r = await api.deleteLastRecording(kind, label);
      setVocab((v) => (v ? { ...v, [mode]: { ...v[mode], recorded: r.recorded } } : v));
      toast.show(r.deleted ? "Deleted the last recording" : "Nothing to delete for this sign");
    } catch (e) {
      toast.show((e as Error).message, "error");
    }
  };

  const addCustom = () => {
    const name = custom.trim();
    if (!name) return;
    // sentences keep their text; the server turns it into a label when the first recording is saved
    const clean = mode === "letters" ? name.toUpperCase() : mode === "sentences" ? name : name.toLowerCase().replace(/\s+/g, "_");
    setVocab((v) =>
      v && !v[mode].suggested.includes(clean)
        ? { ...v, [mode]: { ...v[mode], suggested: [...v[mode].suggested, clean] } }
        : v,
    );
    setLabel(clean);
    setCustom("");
  };

  useEffect(() => {
    let stop = false;
    let timer = 0;
    const poll = async () => {
      try {
        const j = await api.training(kind);
        if (stop) return;
        setJob(j);
        if (j.state === "running") timer = window.setTimeout(poll, 1500);
      } catch {
        /* server offline */
      }
    };
    poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [kind, job?.state === "running"]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log]);

  const train = async () => {
    try {
      setJob(await api.startTraining(kind));
    } catch (e) {
      toast.show((e as Error).message, "error");
    }
  };

  const totalOwn = Object.values(recorded).reduce((a, b) => a + b, 0);
  const totalPublic = Object.values(publicCounts).reduce((a, b) => a + b, 0);
  const readyToTrain = Object.keys({ ...recorded, ...publicCounts }).length >= 2;

  const overlay =
    phase === "countdown" ? (
      <div className="absolute inset-0 grid place-items-center bg-black/40">
        <span className="text-[96px] leading-none font-semibold">{count}</span>
      </div>
    ) : phase === "recording" ? (
      <div className="absolute inset-x-3 bottom-3">
        <div className="flex items-center gap-3 rounded-md bg-black/65 px-3 py-2 text-[13px] backdrop-blur">
          <Circle size={10} className="fill-red-500 text-red-500" aria-hidden />
          <span className="font-medium">Recording {label && display(label)}</span>
          <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/20">
            <span className="block h-full bg-white" style={{ width: `${Math.min(100, (elapsed / SECONDS[mode]) * 100)}%` }} />
          </span>
        </div>
      </div>
    ) : null;

  const jobBadge =
    job?.state === "running" ? <Badge tone="primary">Training</Badge>
    : job?.state === "done" ? <Badge tone="success">Done, model in use</Badge>
    : job?.state === "failed" ? <Badge tone="warning">Failed</Badge>
    : null;

  return (
    <div>
      <PageHeader
        title="Record and train"
        description="Record examples of each sign with the camera, then retrain the model. Only hand-point coordinates are stored, never video."
        actions={
          <Segmented<RecMode>
            label="What to record"
            value={mode}
            onChange={setMode}
            options={[
              { value: "sentences", label: "Sentences" },
              { value: "words", label: "Words" },
              { value: "letters", label: "Letters" },
            ]}
          />
        }
      />

      {error && <div className="mb-4"><Notice tone="error">{error}</Notice></div>}

      <div className="stagger grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
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
              overlay={overlay}
            />
            <div className="flex flex-wrap items-center gap-2 border-t border-border p-3">
              <Button variant="primary" onClick={record} disabled={!cam.on || !label || phase !== "idle" || !sock.connected}>
                <Circle size={12} className="fill-current" aria-hidden />
                {label ? `Record “${display(label)}”` : "Select a sign to record"}
              </Button>
              <Button onClick={undo} disabled={!label || !recorded[label]}>
                <Undo2 size={15} aria-hidden /> Delete last
              </Button>
              <span className="ml-auto text-[12px] text-fg-3">{SECONDS[mode]} s per recording after a 3-second countdown</span>
            </div>
          </Panel>
          <Notice>
            {mode === "sentences"
              ? "Start with your hands down, sign the whole sentence once, then lower your hands. Record each sentence at least 5 times; different people signing it makes the model much better."
              : mode === "letters"
              ? "Hold the sign steady for the whole recording, moving it slightly so the model sees some variety."
              : "Perform the whole sign once, from start to finish. Also record “No sign (idle)” with your hands resting or moving naturally, so the app knows when nobody is signing."}
          </Notice>
        </div>

        <div className="space-y-4">
          <Panel
            title="Signs"
            icon={ListChecks}
            accent="teal"
            actions={<span className="text-[12px] text-fg-3">Target: {GOAL[mode]}+ recordings each</span>}
          >
            <label className="mb-4 flex items-center gap-3 text-[13px]">
              <span className="w-14 shrink-0 text-fg-2">Signer</span>
              <input value={signer} onChange={(e) => setSigner(e.target.value)} placeholder="Your name" className={`${inputClass} w-full`} />
            </label>
            <div
              className={`grid max-h-[320px] gap-1.5 overflow-y-auto pr-1 ${
                mode === "sentences" ? "grid-cols-1" : "grid-cols-3 sm:grid-cols-4 lg:grid-cols-3"
              }`}
            >
              {labels.map((l) => {
                const n = recorded[l] ?? 0;
                const selected = label === l;
                return (
                  <button
                    key={l}
                    onClick={() => setLabel(l)}
                    aria-pressed={selected}
                    className={`flex h-9 items-center justify-between gap-1 rounded-lg px-2.5 text-[13px] font-medium ring-1 ring-inset transition-colors ${
                      selected ? "bg-primary text-on-primary ring-primary" : "bg-surface text-fg ring-border hover:bg-subtle"
                    }`}
                  >
                    <span className="truncate">{display(l)}</span>
                    <span className={`text-[11px] ${selected ? "text-on-primary/80" : n >= GOAL[mode] ? "text-success" : "text-fg-3"}`}>{n}</span>
                  </button>
                );
              })}
            </div>
            <form
              className="mt-3 flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                addCustom();
              }}
            >
              <input
                value={custom}
                onChange={(e) => setCustom(e.target.value)}
                placeholder={
                  mode === "letters" ? "Add a sign, e.g. 0" : mode === "sentences" ? "Add a sentence, e.g. I want to go to the market" : "Add a word, e.g. good luck"
                }
                className={`${inputClass} min-w-0 flex-1`}
              />
              <Button type="submit" disabled={!custom.trim()}>
                <Plus size={15} aria-hidden /> Add
              </Button>
            </form>
          </Panel>

          <Panel title={`Train ${MODE_NAME[mode]} model`} icon={Dumbbell} accent="amber" actions={jobBadge}>
            <dl className="mb-4 grid grid-cols-2 gap-3 text-[13px]">
              <div>
                <dt className="text-fg-3">Your recordings</dt>
                <dd className="text-[16px] font-semibold">{totalOwn.toLocaleString()}</dd>
              </div>
              <div>
                <dt className="text-fg-3">Public samples</dt>
                <dd className="text-[16px] font-semibold">{totalPublic.toLocaleString()}</dd>
              </div>
            </dl>
            <Button variant="primary" className="w-full" onClick={train} disabled={!readyToTrain || job?.state === "running"}>
              <Dumbbell size={15} aria-hidden />
              {job?.state === "running" ? "Training…" : "Start training"}
            </Button>
            {!readyToTrain && <p className="mt-2 text-[12px] text-fg-3">Record at least two different signs first.</p>}
            {job?.state === "failed" && <p className="mt-2 text-[12px] text-danger">Training stopped with an error. See the log below.</p>}
            {job?.log && job.state !== "idle" && (
              <pre
                ref={logRef}
                className="mt-3 max-h-48 overflow-auto rounded-lg bg-stage p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-white/80"
              >
                {job.log}
              </pre>
            )}
          </Panel>
        </div>
      </div>
      {toast.node}
    </div>
  );
}
