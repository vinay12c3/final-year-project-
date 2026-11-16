import { useCallback, useEffect, useRef, useState } from "react";
import { wsUrl, type RoomMessage } from "./api";

export type Presence = { host: number; guest: number };

/**
 * Live link to a conversation room. Reconnects by itself (phones lose the
 * connection when the screen locks). Messages sent here go to everyone else
 * in the room; the sender shows its own copy straight away.
 */
export function useRoom(
  code: string | null,
  role: "host" | "guest",
  handlers: { onMessage: (m: RoomMessage) => void; onHistory?: (ms: RoomMessage[]) => void },
) {
  const [connected, setConnected] = useState(false);
  const [presence, setPresence] = useState<Presence>({ host: 0, guest: 0 });
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const queue = useRef<object[]>([]);
  const h = useRef(handlers);
  h.current = handlers;

  useEffect(() => {
    if (!code) return;
    let stop = false;
    let retry = 0;
    let timer = 0;
    const open = () => {
      const s = new WebSocket(wsUrl(`/ws/room/${encodeURIComponent(code)}?role=${role}`));
      ws.current = s;
      s.onopen = () => {
        retry = 0;
        setConnected(true);
        setError(null);
        for (const m of queue.current.splice(0)) s.send(JSON.stringify(m));
      };
      s.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "history") h.current.onHistory?.(msg.messages);
        else if (msg.type === "message") h.current.onMessage(msg.message);
        else if (msg.type === "presence") setPresence({ host: msg.host, guest: msg.guest });
        else if (msg.type === "error") {
          setError(msg.message);
          stop = true;
        }
      };
      s.onclose = () => {
        setConnected(false);
        if (ws.current === s) ws.current = null;
        if (!stop) timer = window.setTimeout(open, Math.min(8000, 500 * 2 ** retry++));
      };
    };
    open();
    return () => {
      stop = true;
      clearTimeout(timer);
      ws.current?.close();
      ws.current = null;
      setConnected(false);
    };
  }, [code, role]);

  const send = useCallback((who: RoomMessage["who"], text: string, signs?: string[]) => {
    const m = { type: "message", who, text, signs };
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(m));
    else queue.current.push(m);
  }, []);

  return { connected, presence, error, send };
}
