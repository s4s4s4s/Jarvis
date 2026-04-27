"""brain/llama_server.py

Manages a llama-server.exe child process for the duration of the pipeline.
Provides async-safe start / stop / healthcheck helpers.

Usage (from executor or main):

    async with LlamaServerManager() as srv:
        base_url = srv.base_url   # e.g. "http://127.0.0.1:8080"
        ...
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via env or constructor kwargs)
# ---------------------------------------------------------------------------
_DEFAULT_EXE   = r"C:\llama-server\llama-server.exe"
_DEFAULT_MODEL = (
    r"C:\Users\Genn_\.ollama\models\blobs\\"
    r"sha256-eabc98a9bcbfce7fd70f3e07de599f8fda98120fefed5881934161ede8bd1a41"
)
_DEFAULT_HOST  = "127.0.0.1"
_DEFAULT_PORT  = 8080
_DEFAULT_PARALLEL = 4          # --parallel N  (simultaneous inference slots)
_DEFAULT_CTX   = 8192          # --ctx-size per slot
_DEFAULT_GPU_LAYERS = 99       # offload everything to VRAM
_HEALTHCHECK_TIMEOUT  = 120    # seconds to wait for server to become ready
_HEALTHCHECK_INTERVAL = 2      # seconds between health probes


class LlamaServerManager:
    """Context-manager that starts llama-server.exe and stops it on exit."""

    def __init__(
        self,
        exe_path: str = _DEFAULT_EXE,
        model_path: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        parallel: int = _DEFAULT_PARALLEL,
        ctx_size: int = _DEFAULT_CTX,
        n_gpu_layers: int = _DEFAULT_GPU_LAYERS,
    ) -> None:
        self.exe_path    = Path(os.environ.get("LLAMA_SERVER_EXE",   exe_path))
        self.model_path  = Path(os.environ.get("LLAMA_SERVER_MODEL", model_path))
        self.host        = host
        self.port        = port
        self.parallel    = parallel
        self.ctx_size    = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start llama-server.exe synchronously and wait until /health is OK."""
        if not self.exe_path.exists():
            raise FileNotFoundError(f"llama-server not found: {self.exe_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model blob not found: {self.model_path}")

        cmd = [
            str(self.exe_path),
            "--model",       str(self.model_path),
            "--host",        self.host,
            "--port",        str(self.port),
            "--parallel",    str(self.parallel),
            "--ctx-size",    str(self.ctx_size * self.parallel),  # total ctx
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--log-disable",   # suppress noisy server logs
        ]
        logger.info("[LlamaServer] Starting: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # Windows-only
        )
        self._wait_until_ready()
        logger.info("[LlamaServer] Ready at %s (pid=%d)", self.base_url, self._proc.pid)

    def stop(self) -> None:
        """Terminate the server process and free VRAM."""
        if self._proc is None:
            return
        logger.info("[LlamaServer] Stopping pid=%d …", self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None
        logger.info("[LlamaServer] Stopped.")

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------
    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _HEALTHCHECK_TIMEOUT
        url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            # Process died early
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(
                    f"[LlamaServer] Process exited prematurely (rc={self._proc.returncode})"
                )
            try:
                with httpx.Client(timeout=4) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return
            except Exception:
                pass
            time.sleep(_HEALTHCHECK_INTERVAL)
        raise TimeoutError(
            f"[LlamaServer] Did not become ready within {_HEALTHCHECK_TIMEOUT}s"
        )

    async def healthcheck_async(self) -> bool:
        """Non-blocking health probe for use inside async context."""
        url = f"{self.base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "LlamaServerManager":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    async def __aenter__(self) -> "LlamaServerManager":
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.start)
        return self

    async def __aexit__(self, *_) -> None:
        self.stop()
