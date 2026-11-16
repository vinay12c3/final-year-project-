"""Conversation rooms: Person A's laptop and Person B's phone share one live chat.

Rooms live in memory only (nothing is stored), and end when the server stops.
"""
from __future__ import annotations

import secrets
import time

from fastapi import WebSocket

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"     # no 0/O or 1/I
MAX_MESSAGES = 500
ROOM_TTL = 12 * 3600


class Room:
    def __init__(self, code: str):
        self.code = code
        self.created = self.active = time.time()
        self.messages: list[dict] = []
        self.clients: dict[WebSocket, str] = {}     # socket -> role ("host" | "guest")
        self.next_id = 1

    def presence(self) -> dict:
        roles = list(self.clients.values())
        return {"type": "presence", "host": roles.count("host"), "guest": roles.count("guest")}

    async def broadcast(self, msg: dict, skip: WebSocket | None = None):
        for ws in list(self.clients):
            if ws is skip:
                continue
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001  (a closed socket is removed on disconnect)
                pass


class Rooms:
    def __init__(self):
        self.rooms: dict[str, Room] = {}

    def create(self) -> Room:
        self._expire()
        while True:
            code = "".join(secrets.choice(ALPHABET) for _ in range(6))
            if code not in self.rooms:
                self.rooms[code] = Room(code)
                return self.rooms[code]

    def get(self, code: str) -> Room | None:
        return self.rooms.get(code.upper())

    def _expire(self):
        now = time.time()
        for code, r in list(self.rooms.items()):
            if not r.clients and now - r.active > ROOM_TTL:
                del self.rooms[code]

    def add_message(self, room: Room, who: str, text: str, signs: list[str] | None) -> dict:
        m = {"id": room.next_id, "who": who, "text": text[:1000], "time": time.time()}
        if who == "signer":
            m["signs"] = [str(s)[:200] for s in (signs or [])][:30]
        room.next_id += 1
        room.messages = (room.messages + [m])[-MAX_MESSAGES:]
        room.active = time.time()
        return m
