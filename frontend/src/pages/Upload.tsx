import { useEffect, useRef, useState } from "react";
import { Save, Sparkles, UploadCloud, Volume2 } from "lucide-react";
import { Button, ConfidenceRow, Notice, PageHeader, Panel, Segmented, useToast } from "../components/ui";
import { api, type Guess, type Hand, type RecMode, type VideoResult } from "../lib/api";
import { drawHands, speak } from "../lib/camera";

type ImageResult = { hands: Hand[]; top: Guess[]; model_ready: boolean };

function ImagePreview({ src, hands }: { src: string; hands: Hand[] }) {
  const img = useRef<HTMLImageElement | null>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const el = img.current;
    if (!el) return;
    const paint = () => drawHands(canvas.current, hands, el.clientWidth, el.clientHeight);
    if (el.complete) paint();
    el.addEventListener("load", paint);
    window.addEventListener("resize", paint);
    return () => {
      el.removeEventListener("load", paint);
      window.removeEventListener("resize", paint);
    };
  }, [src, hands]);
  return (
    <div className="relative mx-auto w-fit">
      <img ref={img} src={src} alt="Uploaded sign" className="block max-h-[26rem] max-w-full" />
      <canvas ref={canvas} className="pointer-events-none absolute inset-0 h-full w-full" />
    </div>
  );
}

export default function Upload() {
  const [mode, setMode] = useState<RecMode>("sentences");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageRes, setImageRes] = useState<ImageResult | null>(null);
  const [videoRes, setVideoRes] = useState<VideoResult | null>(null);
  const [drag, setDrag] = useState(false);
  const toast = useToast();
  const isVideo = file?.type.startsWith("video/") ?? false;

  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
  }, [url]);

  const choose = (f: File | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/") && !f.type.startsWith("video/")) {
      setError("Choose a photo (JPG, PNG) or a video (MP4, WEBM, MOV).");
      return;
    }
    setError(null);
    setImageRes(null);
    setVideoRes(null);
    setFile(f);
    setUrl(URL.createObjectURL(f));
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      if (isVideo) {
        const r = await api.predictVideo(file, mode);
        if (r.error) setError(r.error);
        setVideoRes(r);
      } else {
        setImageRes(await api.predictImage(file));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const resultText = videoRes?.text || (imageRes?.top[0]?.display ?? "");
  const save = async () => {
    try {
      await api.saveHistory(resultText, isVideo ? mode : "letters", isVideo ? "video" : "image");
      toast.show("Saved to history");
    } catch (e) {
      toast.show((e as Error).message, "error");
    }
  };

  return (
    <div>
      <PageHeader
        title="Photo or video"
        description="Translate a saved file. Photos are read as one letter or number; videos as fingerspelling or words."
      />

      <div className="stagger grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <Panel title="File" icon={UploadCloud} accent="sky" actions={
            file && isVideo ? (
              <Segmented<RecMode>
                label="Video contains"
                value={mode}
                onChange={setMode}
                options={[
                  { value: "sentences", label: "Sentence" },
                  { value: "words", label: "Words" },
                  { value: "letters", label: "Letters" },
                ]}
              />
            ) : null
          }>
          {!url ? (
            <label
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                choose(e.dataTransfer.files[0]);
              }}
              className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-16 text-center transition-colors ${
                drag ? "border-primary bg-primary-soft" : "border-border-strong hover:bg-subtle"
              }`}
            >
              <span className="float grid h-12 w-12 place-items-center rounded-xl bg-primary-soft text-primary">
                <UploadCloud size={18} className="text-fg-2" aria-hidden />
              </span>
              <span>
                <span className="font-semibold text-primary">Click to upload</span>
                <span className="text-fg-2"> or drag and drop</span>
              </span>
              <span className="text-[12px] text-fg-3">JPG, PNG, MP4, WEBM or MOV</span>
              <input type="file" accept="image/*,video/*" className="sr-only" onChange={(e) => choose(e.target.files?.[0])} />
            </label>
          ) : (
            <div className="space-y-4">
              <div className="overflow-hidden rounded-lg bg-stage">
                {isVideo ? <video src={url} controls className="max-h-[26rem] w-full" /> : <ImagePreview src={url} hands={imageRes?.hands ?? []} />}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="primary" onClick={run} disabled={busy}>
                  {busy ? "Translating…" : isVideo ? "Translate video" : "Translate photo"}
                </Button>
                <label className="inline-flex h-9 cursor-pointer items-center rounded-lg px-3.5 font-medium text-fg-2 hover:bg-subtle hover:text-fg">
                  Choose another file
                  <input type="file" accept="image/*,video/*" className="sr-only" onChange={(e) => choose(e.target.files?.[0])} />
                </label>
                <span className="ml-auto truncate text-[12px] text-fg-3">{file?.name}</span>
              </div>
            </div>
          )}
          {error && <div className="mt-4"><Notice tone="error">{error}</Notice></div>}
        </Panel>

        <Panel
          title="Result"
          icon={Sparkles}
          accent="violet"
          actions={
            resultText ? (
              <>
                <Button size="sm" onClick={() => speak(resultText)}>
                  <Volume2 size={15} aria-hidden /> Speak
                </Button>
                <Button size="sm" onClick={save}>
                  <Save size={15} aria-hidden /> Save
                </Button>
              </>
            ) : null
          }
        >
          {!imageRes && !videoRes && <p className="text-fg-3">Upload a file and press Translate to see the result.</p>}

          {imageRes &&
            (!imageRes.hands.length ? (
              <p className="text-fg-2">No hand was found. Use a well-lit photo where the whole hand is visible.</p>
            ) : !imageRes.model_ready ? (
              <p className="text-fg-2">Hands found, but the letters model isn't trained yet.</p>
            ) : (
              <div className="space-y-4">
                <p className="text-[56px] leading-none font-semibold">{imageRes.top[0]?.display}</p>
                <div>
                  <p className="mb-1 text-[12px] font-medium text-fg-3">Top predictions</p>
                  {imageRes.top.map((g, i) => (
                    <ConfidenceRow key={g.label} label={g.display} value={g.confidence} strong={i === 0} />
                  ))}
                </div>
              </div>
            ))}

          {videoRes && !videoRes.error && (
            <div className="space-y-4">
              <p className="text-[24px] leading-snug font-semibold break-words">
                {videoRes.text || <span className="text-fg-3">No confident signs found</span>}
              </p>
              <p className="text-[12px] text-fg-3">
                {videoRes.duration} s analysed, hands visible in{" "}
                {Math.round((videoRes.hand_frames / Math.max(1, videoRes.frames)) * 100)}% of frames
              </p>
              {videoRes.timeline.length > 0 && (
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="text-left text-[12px] text-fg-3">
                      <th className="pb-1 font-medium">Sign</th>
                      <th className="pb-1 text-right font-medium">Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {videoRes.timeline.map((t, i) => (
                      <tr key={i}>
                        <td className="py-1.5 font-medium">{t.display}</td>
                        <td className="py-1.5 text-right text-fg-2">
                          {t.start.toFixed(1)}–{t.end.toFixed(1)} s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {videoRes.clip_top.length > 0 && (
                <div className="border-t border-border pt-3">
                  <p className="mb-1 text-[12px] font-medium text-fg-3">Whole clip as a single word</p>
                  {videoRes.clip_top.slice(0, 3).map((g, i) => (
                    <ConfidenceRow key={g.label} label={g.display} value={g.confidence} strong={i === 0} />
                  ))}
                </div>
              )}
            </div>
          )}
        </Panel>
      </div>
      {toast.node}
    </div>
  );
}
