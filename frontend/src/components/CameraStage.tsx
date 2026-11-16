import { useEffect, useRef, type ReactNode, type RefObject } from "react";
import { Camera, CameraOff } from "lucide-react";
import type { Hand } from "../lib/api";
import { drawHands } from "../lib/camera";
import { HandMark } from "./HandMark";
import { Button } from "./ui";

/** Live mirrored webcam view with the detected hand skeleton drawn on top. */
export function CameraStage({
  videoRef,
  on,
  error,
  hands,
  onStart,
  onStop,
  connected,
  overlay,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  on: boolean;
  error: string | null;
  hands: Hand[];
  onStart: () => void;
  onStop: () => void;
  connected: boolean;
  overlay?: ReactNode;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const v = videoRef.current;
    const natural = v?.videoWidth ? { w: v.videoWidth, h: v.videoHeight } : undefined;
    drawHands(canvasRef.current, on ? hands : [], box.clientWidth, box.clientHeight, natural);
  }, [hands, on, videoRef]);

  return (
    <div className={on ? "live-frame" : ""}>
    <div ref={boxRef} className="relative aspect-[4/3] min-h-[300px] w-full overflow-hidden bg-stage text-white sm:min-h-0">
      <video
        ref={videoRef}
        muted
        playsInline
        className={`absolute inset-0 h-full w-full -scale-x-100 object-cover ${on ? "" : "invisible"}`}
      />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />

      {!on && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 px-6 text-center">
          <HandMark className="float h-20 w-20 opacity-90" />
          <div className="max-w-xs">
            <p className="text-[15px] font-semibold">Camera is off</p>
            <p className="mt-1 text-[13px] leading-relaxed text-white/60">
              Keep both hands and your chest in view, in good light. Video never leaves this computer.
            </p>
          </div>
          <Button variant="primary" onClick={onStart}>
            <Camera size={16} aria-hidden /> Start camera
          </Button>
          {error && <p className="max-w-sm rounded-lg bg-danger px-3 py-2 text-[13px]">{error}</p>}
        </div>
      )}

      {on && overlay}

      {on && (
        <div className="absolute inset-x-3 top-3 flex items-center justify-between gap-2 text-[12px]">
          <span className="flex items-center gap-1.5 rounded-md bg-black/55 px-2 py-1 backdrop-blur">
            <span className={`ping h-2 w-2 rounded-full ${connected ? (hands.length ? "bg-primary" : "bg-fg-3") : "bg-amber-400"}`} aria-hidden />
            {connected ? (hands.length ? `${hands.length} hand${hands.length > 1 ? "s" : ""} detected` : "No hands detected") : "Connecting…"}
          </span>
          <button
            onClick={onStop}
            className="flex items-center gap-1.5 rounded-md bg-black/55 px-2 py-1 backdrop-blur hover:bg-black/75"
          >
            <CameraOff size={14} aria-hidden /> Stop
          </button>
        </div>
      )}
    </div>
    </div>
  );
}
