import { useEffect, useState } from "react";
import { CaseSensitive, GraduationCap, MessagesSquare, Users, Workflow } from "lucide-react";
import { Notice, PageHeader, Panel, Stat } from "../components/ui";
import { api, pct, type Kind, type Metrics } from "../lib/api";

const STEPS = [
  { title: "Capture", text: "The browser reads the webcam and sends compressed frames to the local server." },
  { title: "Pre-process", text: "OpenCV resizes each frame and enhances contrast in low light." },
  { title: "Detect hands", text: "MediaPipe locates 21 three-dimensional landmarks on each hand." },
  { title: "Classify", text: "An MLP reads the hand shape for letters; a bidirectional GRU reads 2 s of motion for words." },
  { title: "Output", text: "Stable predictions are assembled into text that can be saved or spoken." },
];

const TEAM = [
  { name: "K. Pujitha", roll: "23481A12A4" },
  { name: "T. Jaya Sree Fani Priya", roll: "23481A1295" },
  { name: "K. Vinay Manikanta", roll: "23481A12C3" },
  { name: "K. Dheeraj", roll: "23481A12A8" },
];

const FIGURE_NAMES: Record<string, string> = {
  "confusion_matrix.png": "Confusion matrix",
  "model_comparison.png": "Comparison with classic ML",
  "training_curves.png": "Training curves",
  "per_class_f1.png": "F1 score per sign",
  "dataset_distribution.png": "Samples per sign",
};

function Results({ kind, m }: { kind: Kind; m: NonNullable<Metrics[Kind]> }) {
  const rows = Object.entries(m.comparison).sort((a, b) => b[1].accuracy - a[1].accuracy);
  const name = (f: string) => FIGURE_NAMES[f.split("/").pop() ?? ""] ?? f;
  return (
    <Panel title={kind === "static" ? "Letters and numbers model" : kind === "sentence" ? "Sentences model (ISL-CSLTR)" : "Words model"} icon={kind === "static" ? CaseSensitive : MessagesSquare} accent={kind === "static" ? "violet" : "pink"} bodyClassName="">
      <dl className="grid grid-cols-2 divide-border border-b border-border sm:grid-cols-4 sm:divide-x">
        {[
          ["Test accuracy", pct(m.test.accuracy, 1)],
          m.meta.test_top3_accuracy != null ? ["Correct in top 3", pct(m.meta.test_top3_accuracy, 1)] : ["Macro F1", m.test.f1.toFixed(3)],
          ["Classes", String(m.labels.filter((l) => !l.startsWith("_")).length)],
          ["Inference time", `${m.meta.model_latency_ms.toFixed(2)} ms`],
        ].map(([k, v]) => (
          <div key={k} className="px-4 py-3">
            <dt className="text-[12px] text-fg-3">{k}</dt>
            <dd className="text-[22px] font-semibold">{v}</dd>
          </div>
        ))}
      </dl>
      <div className="space-y-4 p-4">
        <p className="text-[13px] text-fg-2">
          Trained {m.meta.trained_at} on {m.meta.n_train.toLocaleString()} samples; validated on{" "}
          {m.meta.n_val.toLocaleString()} and tested on {m.meta.n_test.toLocaleString()} unseen samples.{" "}
          {m.params.toLocaleString()} parameters.
        </p>
        <div className="overflow-x-auto rounded-lg ring-1 ring-border">
          <table className="w-full min-w-[30rem] text-left text-[13px]">
            <thead className="bg-subtle text-[12px] text-fg-3">
              <tr>
                {["Model", "Accuracy", "Precision", "Recall", "F1"].map((h) => (
                  <th key={h} className="px-3 py-2 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map(([n, s]) => (
                <tr key={n} className={n.includes("(ours)") ? "bg-primary-soft/60 font-semibold" : ""}>
                  <td className="px-3 py-2">{n}</td>
                  <td className="px-3 py-2">{pct(s.accuracy, 2)}</td>
                  <td className="px-3 py-2">{s.precision.toFixed(3)}</td>
                  <td className="px-3 py-2">{s.recall.toFixed(3)}</td>
                  <td className="px-3 py-2">{s.f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="grid items-start gap-3 md:grid-cols-2">
          {m.figures.map((f) => (
            <figure key={f} className="rounded-lg p-2 ring-1 ring-border">
              <a href={f} target="_blank" rel="noreferrer">
                <img src={f} alt={name(f)} className="w-full" loading="lazy" />
              </a>
              <figcaption className="px-1 pt-1.5 text-[12px] text-fg-3">{name(f)}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </Panel>
  );
}

export default function About() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.metrics().then(setMetrics).catch((e) => setError(e.message));
  }, []);

  return (
    <div className="stagger space-y-4">
      <PageHeader
        title="About the project"
        description="SignVision converts Indian Sign Language into text in real time with an ordinary webcam, helping deaf and speech-impaired people communicate with people who don't know sign language. It runs entirely on the local computer using open-source tools."
      />

      <Panel title="System pipeline" icon={Workflow} accent="indigo" bodyClassName="">
        <ol className="grid divide-y divide-border sm:grid-cols-5 sm:divide-x sm:divide-y-0">
          {STEPS.map((s, i) => (
            <li key={s.title} className="p-4 transition-colors hover:bg-subtle">
              <span className="grid h-7 w-7 place-items-center rounded-md bg-primary-soft text-[13px] font-semibold text-primary">{i + 1}</span>
              <p className="mt-2 font-semibold">{s.title}</p>
              <p className="mt-1 text-[13px] leading-relaxed text-fg-2">{s.text}</p>
            </li>
          ))}
        </ol>
      </Panel>

      {error && <Notice tone="error">{error}</Notice>}
      {metrics && !metrics.static && !metrics.dynamic && <Notice>No model has been trained yet.</Notice>}
      {metrics?.static && <Results kind="static" m={metrics.static} />}
      {metrics?.dynamic && <Results kind="dynamic" m={metrics.dynamic} />}
      {metrics?.sentence && <Results kind="sentence" m={metrics.sentence} />}

      <div className="grid gap-4 md:grid-cols-[1.5fr_1fr]">
        <Panel title="Team (Batch B12)" icon={Users} accent="teal">
          <dl className="grid gap-3 sm:grid-cols-2">
            {TEAM.map((t) => (
              <Stat key={t.roll} label={t.roll} value={t.name} />
            ))}
          </dl>
        </Panel>
        <Panel title="Project guide" icon={GraduationCap} accent="amber">
          <p className="font-semibold">Ms. Sykam Vani, M.Tech. (Ph.D)</p>
          <p className="mt-0.5 text-[13px] text-fg-2">
            Assistant Professor, Department of Information Technology, Seshadri Rao Gudlavalleru Engineering College
          </p>
        </Panel>
      </div>
    </div>
  );
}
