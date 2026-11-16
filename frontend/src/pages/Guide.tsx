import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { CaseSensitive, Hash, Lightbulb, MessagesSquare, Play, Search, Speech, X } from "lucide-react";
import { Badge, Notice, PageHeader, Panel, Segmented, linkClass } from "../components/ui";

const NUMBERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
const LETTERS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
// ISL signs made with one hand (checked against the photos); all the others use both hands.
const ONE_HAND = new Set(["C", "I", "L", "O", "U", "V", ...NUMBERS]);

type Clip = { label: string; text: string; video: string; poster: string };
type Manifest = { sentences: Clip[]; words: Clip[]; credits?: Record<string, string> };
type Tab = "sentences" | "words" | "letters";
type Filter = "all" | "one" | "two";

/** A sign shown as a looping video: plays while hovered, opens large on click. */
function ClipCard({ clip, onOpen }: { clip: Clip; onOpen: (c: Clip) => void }) {
  const video = useRef<HTMLVideoElement | null>(null);
  const play = () => video.current?.play().catch(() => undefined);
  const stop = () => {
    const v = video.current;
    if (v) {
      v.pause();
      v.currentTime = 0;
    }
  };
  return (
    <li>
      <button
        onClick={() => onOpen(clip)}
        onMouseEnter={play}
        onMouseLeave={stop}
        onFocus={play}
        onBlur={stop}
        className="lift group block w-full overflow-hidden rounded-xl bg-surface text-left ring-1 ring-border hover:ring-primary/60"
      >
        <span className="relative block aspect-video bg-stage">
          <video ref={video} src={clip.video} poster={clip.poster} muted loop playsInline preload="none" className="h-full w-full object-cover" />
          <span className="absolute right-2 bottom-2 grid h-7 w-7 place-items-center rounded-full bg-black/60 text-white transition-opacity group-hover:opacity-0">
            <Play size={13} className="translate-x-px" aria-hidden />
          </span>
        </span>
        <span className="block px-3 py-2.5 text-[14px] leading-snug font-medium text-fg group-hover:text-primary">{clip.text}</span>
      </button>
    </li>
  );
}

function LetterCard({ sign, onOpen }: { sign: string; onOpen: (s: string) => void }) {
  return (
    <li>
      <button
        onClick={() => onOpen(sign)}
        className="lift group block w-full overflow-hidden rounded-xl bg-surface text-left ring-1 ring-border hover:ring-primary/60"
      >
        <img src={`/signs/${sign}.jpg`} alt={`ISL sign for ${sign}`} loading="lazy" className="aspect-square w-full object-cover" />
        <span className="flex items-center justify-between gap-2 border-t border-border px-2.5 py-2">
          <span className="text-[16px] font-semibold text-fg group-hover:text-primary">{sign}</span>
          <span className="text-[11px] text-fg-3">{ONE_HAND.has(sign) ? "1 hand" : "2 hands"}</span>
        </span>
      </button>
    </li>
  );
}

function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div role="dialog" aria-modal="true" aria-label={title} className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4" onClick={onClose}>
      <figure className="page-enter w-full max-w-2xl overflow-hidden rounded-xl bg-surface shadow-2xl ring-1 ring-border" onClick={(e) => e.stopPropagation()}>
        {children}
        <figcaption className="flex items-center justify-between gap-3 px-4 py-3">
          <span className="text-[18px] font-semibold">{title}</span>
          <button onClick={onClose} aria-label="Close" className="grid h-8 w-8 place-items-center rounded-lg text-fg-2 hover:bg-subtle">
            <X size={16} />
          </button>
        </figcaption>
      </figure>
    </div>
  );
}

export default function Guide() {
  const [tab, setTab] = useState<Tab>("sentences");
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [missing, setMissing] = useState(false);
  const [query, setQuery] = useState("");
  const [clip, setClip] = useState<Clip | null>(null);
  const [letter, setLetter] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const keep = (s: string) => filter === "all" || (filter === "one") === ONE_HAND.has(s);

  useEffect(() => {
    fetch("/clips/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setManifest)
      .catch(() => setMissing(true));
  }, []);

  const list = tab === "sentences" ? manifest?.sentences ?? [] : manifest?.words ?? [];
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? list.filter((c) => c.text.toLowerCase().includes(q)) : list;
  }, [list, query]);

  return (
    <div className="stagger space-y-4">
      <PageHeader
        title="Sign guide"
        description={
          <>
            How to sign each <strong className="font-semibold text-fg">Indian Sign Language (ISL)</strong> sentence the app
            understands. Hover over a card to play the sign, and click to watch it large.
          </>
        }
        actions={
          <Segmented<Tab>
            label="Sign type"
            value={tab}
            onChange={(t) => {
              setTab(t);
              setQuery("");
            }}
            options={[
              { value: "sentences", label: `Sentences${manifest ? ` (${manifest.sentences.length})` : ""}` },
              { value: "words", label: `Words${manifest ? ` (${manifest.words.length})` : ""}` },
              { value: "letters", label: "Letters" },
            ]}
          />
        }
      />

      <Panel title="Before you start" icon={Lightbulb} bodyClassName="px-4 py-3">
        <ul className="grid gap-x-8 gap-y-1.5 text-[13px] text-fg-2 md:grid-cols-2">
          <li>Sit far enough back that your head, chest and both hands are in view.</li>
          <li>Face the light; avoid a bright window behind you.</li>
          <li>Sign the whole sentence in one smooth movement, like in the video, then lower your hands.</li>
          <li>
            Need a sentence that isn't here? Record it on <Link to="/record" className={linkClass}>Record and train</Link>.
          </li>
        </ul>
      </Panel>

      {tab !== "letters" && (
        <Panel
          title={tab === "sentences" ? "Sentences" : "Words"}
          icon={tab === "sentences" ? Speech : MessagesSquare}
          actions={
            <label className="relative">
              <Search size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-fg-3" aria-hidden />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={`Search ${tab}…`}
                aria-label={`Search ${tab}`}
                className="h-8 w-56 rounded-lg bg-subtle pr-3 pl-8 text-[13px] text-fg ring-1 ring-border outline-none placeholder:text-fg-3 focus:ring-primary"
              />
            </label>
          }
        >
          {missing ? (
            <Notice tone="warning">
              The sign videos haven't been made yet. Run <code>python -m scripts.make_sign_clips</code> in the backend folder.
            </Notice>
          ) : !manifest ? (
            <p className="text-fg-3">Loading…</p>
          ) : shown.length === 0 ? (
            <p className="py-6 text-center text-fg-3">
              {list.length ? `Nothing matches "${query}".` : `No ${tab} videos yet.`}
            </p>
          ) : (
            <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {shown.map((c) => (
                <ClipCard key={c.label} clip={c} onOpen={setClip} />
              ))}
            </ul>
          )}
          {manifest?.credits && (
            <p className="mt-4 text-[11px] text-fg-3">Videos: {manifest.credits[tab]}</p>
          )}
        </Panel>
      )}

      {tab === "letters" && (
        <>
          <div className="flex justify-end">
            <Segmented<Filter>
              label="Filter letters"
              value={filter}
              onChange={setFilter}
              options={[
                { value: "all", label: "All" },
                { value: "one", label: "One hand" },
                { value: "two", label: "Two hands" },
              ]}
            />
          </div>
          {NUMBERS.some(keep) && (
            <Panel title="Numbers" icon={Hash} actions={<Badge>1–9</Badge>}>
              <ul className="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-9">
                {NUMBERS.filter(keep).map((s) => <LetterCard key={s} sign={s} onOpen={setLetter} />)}
              </ul>
            </Panel>
          )}
          <Panel title="Letters" icon={CaseSensitive} actions={<Badge>A–Z</Badge>}>
            <ul className="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-7">
              {LETTERS.filter(keep).map((s) => <LetterCard key={s} sign={s} onOpen={setLetter} />)}
            </ul>
          </Panel>
        </>
      )}

      {clip && (
        <Modal title={clip.text} onClose={() => setClip(null)}>
          <video src={clip.video} poster={clip.poster} autoPlay muted loop playsInline controls className="aspect-video w-full bg-stage object-contain" />
        </Modal>
      )}
      {letter && (
        <Modal title={`${letter} · ${ONE_HAND.has(letter) ? "one hand" : "both hands"}`} onClose={() => setLetter(null)}>
          <img src={`/signs/${letter}.jpg`} alt={`ISL sign for ${letter}`} className="mx-auto aspect-square w-full max-w-sm object-cover" />
        </Modal>
      )}
    </div>
  );
}
