import { lazy, Suspense, useEffect, useState } from "react";
import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { BookOpen, Database, FileVideo, History as HistoryIcon, Info, Menu, MessagesSquare, Video, X } from "lucide-react";
import { Logo } from "./components/HandMark";
import { api, type Status } from "./lib/api";
import Translate from "./pages/Translate";

const Upload = lazy(() => import("./pages/Upload"));
const Record = lazy(() => import("./pages/Record"));
const History = lazy(() => import("./pages/History"));
const About = lazy(() => import("./pages/About"));
const Guide = lazy(() => import("./pages/Guide"));
const Conversation = lazy(() => import("./pages/Conversation"));
const Room = lazy(() => import("./pages/Room"));

const NAV = [
  { to: "/", label: "Live translation", icon: Video },
  { to: "/conversation", label: "Conversation", icon: MessagesSquare },
  { to: "/guide", label: "Sign guide", icon: BookOpen },
  { to: "/upload", label: "Photo or video", icon: FileVideo },
  { to: "/record", label: "Record and train", icon: Database },
  { to: "/history", label: "History", icon: HistoryIcon },
  { to: "/about", label: "About the project", icon: Info },
];

function ModelStatus() {
  const [status, setStatus] = useState<Status | null>(null);
  const [offline, setOffline] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => {
    api.status().then((s) => {
      setStatus(s);
      setOffline(false);
    }).catch(() => setOffline(true));
  }, [pathname]);

  const row = (name: string, ready?: boolean, count?: number) => (
    <li className="flex items-center justify-between gap-2">
      <span className="flex items-center gap-2 text-fg-2">
        <span className={`h-2 w-2 rounded-full ${ready ? "ping bg-success" : "bg-border-strong"}`} aria-hidden />
        {name}
      </span>
      <span className="text-fg-3">{ready ? `${count} signs` : "not trained"}</span>
    </li>
  );
  return (
    <div className="rounded-lg bg-surface p-3 text-[12px] ring-1 ring-border">
      <p className="mb-2 font-medium text-fg">Models</p>
      {offline ? (
        <p className="text-danger">Server not reachable</p>
      ) : (
        <ul className="space-y-1.5">
          {row("Letters", status?.models.letters.ready, status?.models.letters.labels.length)}
          {row("Words", status?.models.words.ready, status?.models.words.labels.filter((l) => !l.startsWith("_")).length)}
          {row("Sentences", status?.models.sentences?.ready, status?.models.sentences?.labels.length)}
        </ul>
      )}
    </div>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col gap-7 px-3 py-5">
      <NavLink to="/" onClick={onNavigate} className="px-2" aria-label="SignVision home">
        <Logo dark />
      </NavLink>
      <nav aria-label="Main" className="flex flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            onClick={onNavigate}
            className={({ isActive }) =>
              `group relative flex h-9 items-center gap-3 rounded-lg px-2.5 text-[14px] font-medium transition-colors duration-150 ${
                isActive ? "bg-surface text-fg" : "text-fg-2 hover:bg-surface/60 hover:text-fg"
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && <span className="absolute top-2 bottom-2 -left-3 w-[3px] rounded-r-full bg-primary" aria-hidden />}
                <Icon size={17} aria-hidden className={`transition-colors ${isActive ? "text-primary" : "text-fg-3 group-hover:text-fg-2"}`} />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto space-y-3">
        <ModelStatus />
        <p className="px-1 text-[12px] leading-relaxed text-fg-3">
          Batch B12, Dept. of IT
          <br />
          Seshadri Rao Gudlavalleru Engineering College
        </p>
      </div>
    </div>
  );
}

function Shell() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const sidebarBg = "bg-sidebar border-r border-border";
  return (
    <div className="relative min-h-screen lg:pl-[256px]">

      <aside className={`fixed inset-y-0 left-0 z-20 hidden w-[256px] overflow-y-auto lg:block ${sidebarBg}`}>
        <Sidebar />
      </aside>

      <div className={`sticky top-0 z-30 flex h-14 items-center justify-between px-4 lg:hidden ${sidebarBg}`}>
        <Logo dark />
        <button
          className="grid h-9 w-9 place-items-center rounded-lg text-fg ring-1 ring-border-strong"
          aria-label="Open menu"
          aria-expanded={open}
          onClick={() => setOpen(true)}
        >
          <Menu size={18} />
        </button>
      </div>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <div className={`page-enter absolute inset-y-0 left-0 w-[280px] shadow-2xl ${sidebarBg}`}>
            <button
              className="absolute top-5 right-3 grid h-8 w-8 place-items-center rounded-lg text-fg-2 hover:bg-surface"
              aria-label="Close menu"
              onClick={() => setOpen(false)}
            >
              <X size={18} />
            </button>
            <Sidebar onNavigate={() => setOpen(false)} />
          </div>
        </div>
      )}

      <main className="relative z-10 min-w-0 px-4 py-6 sm:px-8 lg:py-8">
        <div key={location.pathname} className="page-enter mx-auto max-w-[1180px]">
          <Suspense fallback={<p className="py-10 text-fg-3">Loading…</p>}>
            <Routes location={location}>
              <Route path="/" element={<Translate />} />
              <Route path="/conversation" element={<Conversation />} />
              <Route path="/guide" element={<Guide />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/record" element={<Record />} />
              <Route path="/history" element={<History />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<Translate />} />
            </Routes>
          </Suspense>
        </div>
      </main>
    </div>
  );
}

/** Person B's room page has no sidebar: it is opened on their own phone. */
function Root() {
  const location = useLocation();
  if (location.pathname.startsWith("/room/"))
    return (
      <Suspense fallback={<p className="p-6 text-fg-3">Loading…</p>}>
        <Routes>
          <Route path="/room/:code" element={<Room />} />
        </Routes>
      </Suspense>
    );
  return <Shell />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Root />
    </BrowserRouter>
  );
}
