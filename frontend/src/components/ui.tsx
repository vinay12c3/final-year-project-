import { useEffect, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { AlertCircle, CheckCircle2, Info, type LucideIcon } from "lucide-react";
import type { Mode } from "../lib/api";

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" }) {
  const styles: Record<Variant, string> = {
    primary: "bg-primary text-on-primary font-semibold hover:bg-primary-hover",
    secondary: "bg-subtle text-fg ring-1 ring-inset ring-border-strong hover:bg-border",
    ghost: "text-fg-2 hover:bg-subtle hover:text-fg",
    danger: "bg-subtle text-danger ring-1 ring-inset ring-border-strong hover:bg-danger-soft",
  };
  const sizes = { sm: "h-8 px-2.5 text-[13px] gap-1.5", md: "h-9 px-3.5 text-[14px] gap-2" };
  return (
    <button
      className={`press inline-flex shrink-0 items-center justify-center rounded-lg font-medium transition-all
        disabled:pointer-events-none disabled:opacity-50 ${sizes[size]} ${styles[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/** Two-option segmented control. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; hint?: string }[];
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg bg-surface p-0.5 ring-1 ring-inset ring-border">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          title={o.hint}
          onClick={() => onChange(o.value)}
          className={`h-8 rounded-md px-3 text-[13px] font-medium transition-all duration-300 ${
            value === o.value ? "bg-primary text-on-primary font-semibold" : "text-fg-2 hover:text-fg"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ModeSwitch({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <Segmented
      label="Recognition mode"
      value={mode}
      onChange={onChange}
      options={[
        { value: "letters", label: "Letters", hint: "Fingerspelling A–Z and numbers 1–9" },
        { value: "words", label: "Words", hint: "Moving signs such as hello, thank you" },
      ]}
    />
  );
}

export type Accent = "indigo" | "violet" | "pink" | "teal" | "amber" | "sky";
export const ACCENT_CHIP: Record<Accent, string> = {
  indigo: "bg-primary-soft text-primary",
  violet: "bg-primary-soft text-primary",
  pink: "bg-primary-soft text-primary",
  teal: "bg-primary-soft text-primary",
  amber: "bg-primary-soft text-primary",
  sky: "bg-primary-soft text-primary",
};

/** Rounded card with an optional header row and coloured icon. */
export function Panel({
  title,
  icon: Icon,
  accent = "indigo",
  actions,
  children,
  className = "",
  bodyClassName = "p-4",
}: {
  title?: ReactNode;
  icon?: LucideIcon;
  accent?: Accent;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={`overflow-hidden rounded-xl bg-surface ring-1 ring-border ${className}`}>
      {(title || actions) && (
        <header className="flex min-h-12 flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2">
          {title && (
            <h2 className="flex items-center gap-2 text-[14px] font-semibold text-fg">
              {Icon && (
                <span className={`grid h-6 w-6 place-items-center rounded-md ${ACCENT_CHIP[accent]}`}>
                  <Icon size={14} aria-hidden />
                </span>
              )}
              {title}
            </h2>
          )}
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <h1 className="text-[24px] leading-8 font-semibold tracking-[-0.01em] text-fg">{title}</h1>
        {description && <p className="mt-1 text-[14px] text-fg-2">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function ConfidenceRow({ label, value, strong = false }: { label: string; value: number; strong?: boolean }) {
  return (
    <div className="grid grid-cols-[4.5rem_1fr_2.75rem] items-center gap-3 py-1.5 text-[13px]">
      <span className={`truncate ${strong ? "font-semibold text-fg" : "text-fg-2"}`}>{label}</span>
      <span className="h-1.5 overflow-hidden rounded-full bg-subtle">
        <span
          className={`block h-full rounded-full transition-[width] duration-300 ${strong ? "bar-fill" : "bg-border-strong"}`}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      </span>
      <span className="text-right text-fg-2">{Math.round(value * 100)}%</span>
    </div>
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "neutral" | "success" | "warning" | "primary"; children: ReactNode }) {
  const styles = {
    neutral: "bg-canvas text-fg-2 ring-border",
    success: "bg-success-soft text-success ring-success/20",
    warning: "bg-warning-soft text-warning ring-warning/20",
    primary: "bg-primary-soft text-primary ring-primary/20",
  };
  return (
    <span className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-[12px] font-medium ring-1 ring-inset ${styles[tone]}`}>
      {children}
    </span>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "error" | "warning" }) {
  const styles = {
    info: "bg-primary-soft text-fg ring-primary/15",
    warning: "bg-warning-soft text-fg ring-warning/20",
    error: "bg-danger-soft text-danger ring-danger/20",
  };
  const Icon = tone === "error" ? AlertCircle : Info;
  return (
    <div className={`flex gap-2.5 rounded-lg px-3.5 py-3 text-[13px] leading-relaxed ring-1 ring-inset ${styles[tone]}`}>
      <Icon size={16} className={`mt-0.5 shrink-0 ${tone === "error" ? "" : tone === "warning" ? "text-warning" : "text-primary"}`} aria-hidden />
      <div>{children}</div>
    </div>
  );
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <dt className="text-[12px] text-fg-3">{label}</dt>
      <dd className="text-[14px] font-semibold text-fg">{value}</dd>
    </div>
  );
}

/** Short-lived confirmation in the bottom-right corner. */
export function useToast() {
  const [msg, setMsg] = useState<{ text: string; tone: "ok" | "error"; id: number } | null>(null);
  useEffect(() => {
    if (!msg) return;
    const t = setTimeout(() => setMsg(null), msg.tone === "error" ? 6000 : 2600);
    return () => clearTimeout(t);
  }, [msg]);
  const show = (text: string, tone: "ok" | "error" = "ok") => setMsg({ text, tone, id: Date.now() });
  const node = msg ? (
    <div
      key={msg.id}
      role="status"
      className="toast-in fixed right-4 bottom-4 z-50 flex max-w-[calc(100vw-2rem)] items-start gap-2.5 rounded-lg bg-subtle px-4 py-3 text-[13px] shadow-xl shadow-black/40 ring-1 ring-border-strong sm:max-w-sm"
    >
      {msg.tone === "error" ? (
        <AlertCircle size={16} className="mt-0.5 shrink-0 text-danger" aria-hidden />
      ) : (
        <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-success" aria-hidden />
      )}
      <span className="text-fg">{msg.text}</span>
    </div>
  ) : null;
  return { show, node };
}

export const linkClass = "font-medium text-primary hover:underline underline-offset-2";
