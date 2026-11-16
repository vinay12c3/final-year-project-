// Types and small helpers for talking to the FastAPI backend.

export type Mode = "letters" | "words";
export type Kind = "static" | "dynamic" | "sentence";
/** The three recognisers: letters (static), words (dynamic) and whole sentences. */
export type RecMode = Mode | "sentences";
export const kindOf = (m: RecMode): Kind => (m === "letters" ? "static" : m === "words" ? "dynamic" : "sentence");

export type Hand = { side: string; score: number; points: [number, number][] };
export type Guess = { label: string; display: string; confidence: number };

export type LiveMode = Mode | "sentences";
export type FrameMessage = {
  type: "frame";
  mode: LiveMode;
  signing?: number;
  typed_text?: string;
  hands: Hand[];
  current: Guess | null;
  top: Guess[];
  progress: number;
  typed: string | null;
  text: string;
  model_ready: boolean;
  timing: { detect_ms: number; model_ms: number; total_ms: number };
};
export type TextMessage = { type: "text"; text: string; mode: LiveMode };

export type ModelStatus = {
  ready: boolean;
  labels: string[];
  trained_at: string | null;
  test_accuracy: number | null;
};
export type Status = {
  models: Record<Mode, ModelStatus> & { sentences?: ModelStatus & { texts: Record<string, string> } };
  hand_model: boolean;
  speech?: { model: string; downloaded: boolean };
};

/** One recognised sign in a conversation: a word sign or a fingerspelled word. */
export type SignToken =
  | { kind: "word"; label: string }
  | { kind: "spell"; text: string }
  | { kind: "sentence"; label: string; text: string };
export type Phrases = Record<string, { text: string; kind: "greeting" | "question" | "statement" | "polite" }>;
export type Transcript = { text: string; seconds: number; audio_seconds: number };

export type HistoryItem = { id: number; text: string; mode: RecMode | "conversation"; source: string; created_at: string };

export type Vocabulary = Record<
  RecMode,
  { suggested: string[]; recorded: Record<string, number>; public: Record<string, number>; texts?: Record<string, string> }
>;

export type TrainJob = {
  kind: Kind;
  state: "idle" | "running" | "done" | "failed";
  log: string;
  returncode?: number | null;
};

export type Scores = { accuracy: number; precision: number; recall: number; f1: number; top3_accuracy?: number };
export type Metrics = Partial<
  Record<
    Kind,
    {
      title: string;
      labels: string[];
      test: Scores;
      comparison: Record<string, Scores & { train_seconds?: number }>;
      figures: string[];
      meta: { n_train: number; n_val: number; n_test: number; model_latency_ms: number; trained_at: string; test_top3_accuracy?: number };
      params: number;
    }
  >
>;

export type RoomMessage = { id: number; who: "signer" | "speaker"; text: string; signs?: string[] };
export type Share = { state: "off" | "downloading" | "starting" | "on" | "failed"; url: string | null; error: string | null };

const audioName = (a: Blob) => (a.type.includes("ogg") ? "speech.ogg" : a.type.includes("mp4") ? "speech.m4a" : "speech.webm");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error("Can't reach the SignVision server. Start it with run.bat (or run.sh) and reload.");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => request<Status>("/api/status"),
  vocabulary: () => request<Vocabulary>("/api/vocabulary"),
  metrics: () => request<Metrics>("/api/metrics"),
  history: () => request<HistoryItem[]>("/api/history"),
  saveHistory: (text: string, mode: RecMode | "conversation", source: string) =>
    request<HistoryItem>("/api/history", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode, source }),
    }),
  deleteHistory: (id: number) => request<{ deleted: number }>(`/api/history/${id}`, { method: "DELETE" }),
  clearHistory: () => request<{ deleted: number }>("/api/history", { method: "DELETE" }),
  deleteLastRecording: (kind: Kind, label: string) =>
    request<{ deleted: boolean; recorded: Record<string, number> }>(
      `/api/recordings/${kind}/${encodeURIComponent(label)}/last`,
      { method: "DELETE" },
    ),
  startTraining: (kind: Kind) => request<TrainJob>(`/api/train/${kind}`, { method: "POST" }),
  training: (kind: Kind) => request<TrainJob>(`/api/train/${kind}`),
  compose: (tokens: SignToken[]) =>
    request<{ sentence: string }>("/api/conversation/compose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tokens }),
    }),
  phrases: () => request<Phrases>("/api/conversation/phrases"),
  transcribe: (audio: Blob) => {
    const body = new FormData();
    body.append("audio", audio, audioName(audio));
    return request<Transcript>("/api/transcribe", { method: "POST", body });
  },
  roomTranscribe: (code: string, audio: Blob) => {
    const body = new FormData();
    body.append("audio", audio, audioName(audio));
    return request<Transcript>(`/api/rooms/${encodeURIComponent(code)}/transcribe`, { method: "POST", body });
  },
  createRoom: () => request<{ code: string }>("/api/rooms", { method: "POST" }),
  room: (code: string) => request<{ code: string; messages: RoomMessage[]; host: number; guest: number }>(`/api/rooms/${encodeURIComponent(code)}`),
  share: () => request<Share>("/api/share"),
  shareStart: () => request<Share>("/api/share/start", { method: "POST" }),
  shareStop: () => request<Share>("/api/share/stop", { method: "POST" }),
  predictImage: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ hands: Hand[]; top: Guess[]; model_ready: boolean }>("/api/predict/image", {
      method: "POST",
      body,
    });
  },
  predictVideo: (file: File, mode: RecMode) => {
    const body = new FormData();
    body.append("file", file);
    body.append("mode", mode);
    return request<VideoResult>("/api/predict/video", { method: "POST", body });
  },
};

export type VideoResult = {
  error?: string;
  duration: number;
  frames: number;
  hand_frames: number;
  text: string;
  timeline: { label: string; display: string; start: number; end: number; confidence?: number }[];
  clip_top: Guess[];
};

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}`;
}

export const pct = (x: number | null | undefined, digits = 0) =>
  x == null ? "–" : `${(x * 100).toFixed(digits)}%`;
