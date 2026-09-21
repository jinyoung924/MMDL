"""Run metadata: versions, GPU, peak VRAM sampler, git commit."""
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from importlib.metadata import PackageNotFoundError, version


def pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def nvidia_smi(query: str) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]


def git_commit(repo_dir: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


class VramSampler:
    """Polls nvidia-smi (device-wide, so it also covers vLLM's engine subprocess)."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            vals = nvidia_smi("memory.used")
            if vals:
                try:
                    self.peak_mib = max(self.peak_mib, int(float(vals[0])))
                except ValueError:
                    pass
            self._stop.wait(self.interval)

    def start(self):
        self._t.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        self._t.join(timeout=5)
        return self.peak_mib


def environment_info(repo_dir: str) -> dict:
    gpu = nvidia_smi("name,memory.total,driver_version")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": pkg_version("torch"),
        "vllm": pkg_version("vllm"),
        "transformers": pkg_version("transformers"),
        "datasets": pkg_version("datasets"),
        "gpu": gpu[0] if gpu else None,
        "git_commit": git_commit(repo_dir),
        "hf_home": os.environ.get("HF_HOME"),
    }


def write_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def now() -> float:
    return time.time()
