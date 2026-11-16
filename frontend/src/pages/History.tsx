import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Copy, History as HistoryIcon, Trash2, Volume2 } from "lucide-react";
import { Badge, Button, Notice, PageHeader, Panel, linkClass, useToast } from "../components/ui";
import { api, type HistoryItem } from "../lib/api";
import { speak } from "../lib/camera";

const SOURCE: Record<string, string> = { camera: "Live camera", image: "Photo", video: "Video", conversation: "Conversation" };

function when(iso: string) {
  return new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export default function History() {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  useEffect(() => {
    api.history().then(setItems).catch((e) => setError(e.message));
  }, []);

  const remove = async (id: number) => {
    await api.deleteHistory(id);
    setItems((xs) => xs?.filter((x) => x.id !== id) ?? null);
  };

  const clearAll = async () => {
    if (!confirm("Delete every saved translation? This can't be undone.")) return;
    await api.clearHistory();
    setItems([]);
    toast.show("History cleared");
  };

  return (
    <div>
      <PageHeader
        title="History"
        description="Translations saved from the camera, photos and videos."
        actions={
          items && items.length > 0 ? (
            <Button variant="danger" onClick={clearAll}>
              <Trash2 size={15} aria-hidden /> Clear history
            </Button>
          ) : null
        }
      />

      {error && <Notice tone="error">{error}</Notice>}

      {items && (
        <Panel title="Saved translations" icon={HistoryIcon} accent="pink" bodyClassName="">
          {items.length === 0 ? (
            <div className="px-6 py-14 text-center">
              <p className="font-semibold">No saved translations</p>
              <p className="mt-1 text-fg-2">
                Translate something on <Link to="/" className={linkClass}>Live translation</Link> and press Save.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-left">
                <thead className="border-b border-border bg-subtle text-[12px] text-fg-3">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">Text</th>
                    <th className="px-4 py-2.5 font-medium">Source</th>
                    <th className="px-4 py-2.5 font-medium">Saved</th>
                    <th className="px-4 py-2.5" aria-label="Actions" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {items.map((it) => (
                    <tr key={it.id} className="hover:bg-subtle/60">
                      <td className="px-4 py-3 text-[15px] font-medium break-words whitespace-pre-line">{it.text}</td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className="flex items-center gap-1.5 text-fg-2">
                          {SOURCE[it.source] ?? it.source} <Badge>{it.mode}</Badge>
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-fg-2">{when(it.created_at)}</td>
                      <td className="px-4 py-3">
                        <div className="flex justify-end gap-1">
                          <Button size="sm" variant="ghost" aria-label="Speak" title="Speak" onClick={() => speak(it.text)}>
                            <Volume2 size={15} aria-hidden />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            aria-label="Copy"
                            title="Copy"
                            onClick={() => navigator.clipboard?.writeText(it.text).then(() => toast.show("Copied to clipboard"))}
                          >
                            <Copy size={15} aria-hidden />
                          </Button>
                          <Button size="sm" variant="ghost" aria-label="Delete" title="Delete" onClick={() => remove(it.id)} className="hover:text-danger">
                            <Trash2 size={15} aria-hidden />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      )}
      {toast.node}
    </div>
  );
}
