"""Share the local server over the internet with a Cloudflare Quick Tunnel.

`cloudflared tunnel --url http://localhost:8000` gives a temporary
https://<random>.trycloudflare.com address without an account. It lasts
while the process runs. HTTPS also lets phones use their microphone.
"""
from __future__ import annotations

import os
import platform
import re
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .config import BACKEND_DIR

VERSION = "2026.10.7"
BIN_DIR = BACKEND_DIR / "assets" / "cloudflared"
_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _asset() -> str:
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if system == "windows":
        return f"cloudflared-windows-{arch}.exe"
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz"
    return f"cloudflared-linux-{arch}"


def binary_path():
    return BIN_DIR / ("cloudflared.exe" if platform.system() == "Windows" else "cloudflared")


def is_downloaded() -> bool:
    return binary_path().exists()


def download(progress=print) -> None:
    if is_downloaded():
        return
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    asset = _asset()
    if asset.endswith(".tgz"):
        raise RuntimeError("On macOS install it with:  brew install cloudflared")
    tmp = binary_path().with_suffix(".part")
    for url in (f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/{asset}",
                f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"):
        try:
            progress(f"downloading {url}")
            urllib.request.urlretrieve(url, tmp)
            break
        except Exception as e:  # noqa: BLE001
            progress(f"  failed: {e}")
    else:
        raise RuntimeError("Could not download cloudflared. Check the internet connection.")
    tmp.chmod(tmp.stat().st_mode | stat.S_IEXEC)
    os.replace(tmp, binary_path())


class Tunnel:
    """One background cloudflared process, started on demand from the web app."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None
        self.error: str | None = None
        self.state = "off"                 # off | downloading | starting | on | failed
        self._lock = threading.Lock()

    def status(self) -> dict:
        if self.proc is not None and self.proc.poll() is not None and self.state in ("starting", "on"):
            self.state, self.url = "failed", None
            self.error = self.error or "The tunnel stopped. Check the internet connection and try again."
        return {"state": self.state, "url": self.url if self.state == "on" else None, "error": self.error, "downloaded": is_downloaded()}

    def start(self) -> dict:
        with self._lock:
            if self.state in ("downloading", "starting", "on") and (self.proc is None or self.proc.poll() is None):
                return self.status()
            self.state, self.url, self.error = "starting", None, None
            threading.Thread(target=self._run, daemon=True).start()
        return self.status()

    def _run(self):
        try:
            if not is_downloaded():
                self.state = "downloading"
                download(progress=lambda _m: None)
            self.state = "starting"
            flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            self.proc = subprocess.Popen(
                [str(binary_path()), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                creationflags=flags,
            )
            deadline = time.time() + 60
            for line in self.proc.stdout:            # cloudflared prints the address in its log
                m = _URL_RE.search(line)
                if m and not self.url:
                    url = self.url = m.group(0)
                    threading.Thread(target=self._wait_until_reachable, args=(url,), daemon=True).start()
                if self.state == "starting" and not self.url and time.time() > deadline:
                    self.error = "The tunnel did not start in time."
                    self.stop()
                    break
            if self.state == "starting":
                self.state = "failed"
                self.error = self.error or "Could not start the tunnel. Check the internet connection."
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "failed", str(e)

    def _wait_until_reachable(self, url: str):
        """A new trycloudflare name takes a few seconds to appear in DNS; only report 'on' once it answers."""
        import socket
        host = url.split("//", 1)[1]
        end = time.time() + 90
        while time.time() < end and self.url == url:
            try:
                socket.getaddrinfo(host, 443)
                with urllib.request.urlopen(f"{url}/favicon.svg", timeout=10) as r:
                    if r.status == 200:
                        self.state = "on"
                        return
            except urllib.error.HTTPError:            # any answer from Cloudflare means the name works
                self.state = "on"
                return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        if self.url == url and self.state == "starting":
            self.state, self.error = "on", None      # let the user try anyway

    def stop(self) -> dict:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.state, self.url, self.error = "off", None, None
        return self.status()
