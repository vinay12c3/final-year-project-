import { HAND_CONNECTIONS } from "../lib/camera";

// An open palm drawn as the same 21-point skeleton the recogniser sees.
const PALM: [number, number][] = [
  [50, 92], // 0 wrist
  [37, 85], [27, 76], [20, 66], [15, 57], // thumb
  [39, 58], [37, 44], [36, 35], [35, 26], // index
  [50, 56], [50, 41], [50, 31], [50, 21], // middle
  [61, 58], [63, 45], [64, 36], [65, 28], // ring
  [70, 64], [75, 54], [78, 47], [80, 40], // little
];

export function HandMark({ className = "", dot = "#2DD4BF", bone = "rgba(255,255,255,0.45)" }: { className?: string; dot?: string; bone?: string }) {
  return (
    <svg viewBox="8 14 80 84" className={className} aria-hidden>
      {HAND_CONNECTIONS.map(([a, b]) => (
        <line key={`${a}-${b}`} x1={PALM[a][0]} y1={PALM[a][1]} x2={PALM[b][0]} y2={PALM[b][1]}
          stroke={bone} strokeWidth={2} strokeLinecap="round" />
      ))}
      {PALM.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i % 4 === 0 && i > 0 ? 3.4 : 2.6} fill={dot} />
      ))}
    </svg>
  );
}

export function Logo({ dark = false }: { dark?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary">
        <HandMark className="h-6 w-6" dot="#042f2b" bone="rgba(4,47,43,0.55)" />
      </span>
      <span className={`text-[16px] font-bold tracking-[-0.01em] text-fg`}>SignVision</span>
    </span>
  );
}
