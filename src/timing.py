# src/timing.py
import time
from typing import Dict, Optional


class Timer:
    def __init__(self):
        self._t: Dict[str, Dict[str, Optional[float]]] = {}

    def start(self, name: str) -> None:
        self._t[name] = {"start": time.time(), "end": None}

    def stop(self, name: str) -> None:
        if name in self._t and self._t[name]["end"] is None:
            self._t[name]["end"] = time.time()

    def seconds(self, name: str) -> Optional[float]:
        if name not in self._t:
            return None
        end = self._t[name]["end"] or time.time()
        return end - self._t[name]["start"]

    def summary(self) -> Dict[str, float]:
        out = {}
        for k in self._t:
            out[k] = round(self.seconds(k) or 0.0, 3)
        return out
