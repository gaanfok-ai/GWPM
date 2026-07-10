from __future__ import annotations

import sys
import time


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class ProgressPrinter:
    """Single-line terminal progress for long row-by-row API extraction."""

    def __init__(self, total: int, update_every: int) -> None:
        self.total = total
        self.update_every = update_every
        self.started_at = time.monotonic()

    def maybe_update(self, processed: int, current_id: str | None) -> None:
        if self.update_every <= 0:
            return
        if processed != self.total and processed % self.update_every != 0:
            return
        self.print(processed=processed, current_id=current_id, newline=processed == self.total)

    def print(self, processed: int, current_id: str | None, newline: bool = False) -> None:
        elapsed = time.monotonic() - self.started_at
        pct = processed / self.total * 100 if self.total else 100
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (self.total - processed) / rate if rate > 0 else 0
        message = (
            f"Progress {processed}/{self.total} ({pct:5.1f}%) | "
            f"elapsed {format_duration(elapsed)} | "
            f"eta {format_duration(remaining)} | "
            f"well {current_id or '-'}"
        )
        print(f"\r{message}", end="\n" if newline else "", file=sys.stderr, flush=True)

