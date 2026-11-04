"""Shared helpers: result files, timing and simple statistics."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / 'docs' / 'experiments'


def save(name, payload, merge=False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {'experiment': name, 'python': platform.python_version(),
               'numpy': np.__version__, **payload}
    path = RESULTS / f'{name}.json'
    if merge and path.exists():
        # Re-running one section (--only) keeps the other sections of the file.
        payload = {**json.loads(path.read_text(encoding='utf-8')), **payload}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'-> {path.relative_to(RESULTS.parents[1])}')
    return path


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
