"""Shared logging helper for runner / analysis scripts.

Goals
-----
1. Every runner script (run_phaseB_smoke, run_phaseD_analysis, ...)
   writes a full copy of its stdout + stderr to a log file in the
   results folder, with a meaningful filename.
2. The script always prints a final one-line status — [DONE] on success,
   [ERROR] on any unhandled exception — that points at the log file.
3. The helper is opt-in: a script that doesn't import it keeps working
   exactly as before.

Usage from a runner script
--------------------------
    from experiments._run_logger import run_with_logging, set_log_path

    def main() -> int:
        args = parse_args()
        ...
        # As soon as we know where the run's output goes, register
        # a log path. Path is created on first write.
        set_log_path(
            Path(args.out) / "logs"
            / f"run_phaseB_smoke_{utc_stamp()}.txt"
        )
        ...

    if __name__ == "__main__":
        sys.exit(run_with_logging(main, script_name="run_phaseB_smoke"))

If ``set_log_path`` is never called, the helper falls back to
``results/_logs/<script>_<timestamp>.txt`` so a log file is always
produced.

Implementation notes
--------------------
* Stdout and stderr are wrapped with a Tee that buffers in memory until
  set_log_path() opens a real file, then flushes the buffer and
  redirects writes to that file. This lets the script defer the
  log-path decision until after argparse runs.
* The terminal stream still receives every write in real time, so the
  user sees the same output they always did.
* A KeyboardInterrupt or any other exception inside main() is caught
  for the termination line and then re-raised so the shell sees the
  right exit code (130 for SIGINT, 1 for unhandled exceptions).
* SystemExit is allowed through; if the code is non-zero we treat it
  as an error termination.
"""

from __future__ import annotations

import datetime as _dt
import io
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Union


# ---------------------------------------------------------------------
# Module state — there is only ever one active logger per process.
# ---------------------------------------------------------------------

_buffer: Optional[io.StringIO] = None
_log_handle = None  # type: Optional[io.TextIOWrapper]
_log_path: Optional[Path] = None
_orig_stdout = None
_orig_stderr = None


# ---------------------------------------------------------------------
# Tee
# ---------------------------------------------------------------------

class _Tee:
    """File-like object that fan-outs writes to two streams.

    The first stream is the user's terminal; the second is either an
    in-memory buffer (before set_log_path is called) or the open log
    file (after).
    """

    def __init__(self, terminal, second):
        self.terminal = terminal
        self.second = second

    def write(self, data: str) -> int:  # noqa: D401 - file-like API
        try:
            self.terminal.write(data)
        except Exception:
            pass
        try:
            self.second.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        for s in (self.terminal, self.second):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        try:
            return self.terminal.isatty()
        except Exception:
            return False

    def fileno(self):  # noqa: D401 - file-like API
        # Some libraries probe fileno() to detect a real tty; defer to
        # the terminal stream so that detection still works.
        return self.terminal.fileno()


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------

def utc_stamp() -> str:
    """Compact UTC timestamp suitable for filenames."""
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def set_log_path(path: Union[Path, str]) -> Path:
    """Register the log file the runner wants its output written to.

    Idempotent: only the first call in a process takes effect; later
    calls are silently ignored so import-driven sub-runs (e.g. the
    post-action analyzer chained from run_phaseB_smoke) cannot
    override the parent run's log path.
    """
    global _log_handle, _log_path, _buffer
    if _log_handle is not None:
        return _log_path  # type: ignore[return-value]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "w", encoding="utf-8", buffering=1)
    if _buffer is not None:
        fh.write(_buffer.getvalue())
        fh.flush()
    _log_handle = fh
    _log_path = p
    # Swap the Tee's second stream from the in-memory buffer to the
    # file handle so subsequent writes go straight to disk.
    for attr in ("stdout", "stderr"):
        cur = getattr(sys, attr)
        if isinstance(cur, _Tee) and cur.second is _buffer:
            cur.second = fh
    return p


def current_log_path() -> Optional[Path]:
    """Return the active log path, if any."""
    return _log_path


# ---------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------

@contextmanager
def _capture() -> Iterator[None]:
    global _buffer, _log_handle, _log_path, _orig_stdout, _orig_stderr
    _buffer = io.StringIO()
    _log_handle = None
    _log_path = None
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, _buffer)
    sys.stderr = _Tee(_orig_stderr, _buffer)
    try:
        yield
    finally:
        # Restore the original streams; close the file handle if any.
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        if _log_handle is not None:
            try:
                _log_handle.flush()
                _log_handle.close()
            except Exception:
                pass


def _format_elapsed(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


def _fallback_log_path(script_name: str) -> Path:
    """Where to write the log if the script never called set_log_path."""
    return Path("results") / "_logs" / f"{script_name}_{utc_stamp()}.txt"


def run_with_logging(
    main_fn: Callable[[], int],
    script_name: str,
    default_log_path: Optional[Callable[[], Union[Path, str]]] = None,
) -> int:
    """Run ``main_fn()`` under stdout/stderr capture and emit a
    [DONE]/[ERROR] line + log path at the end.

    Returns the integer exit code the script should propagate to the
    shell. Caller usually wraps this in ``sys.exit(...)``.
    """
    started = _dt.datetime.utcnow()
    exit_code = 0
    error_info: Optional[tuple] = None

    with _capture():
        try:
            rc = main_fn()
            exit_code = int(rc) if rc is not None else 0
            if exit_code != 0:
                error_info = ("ExitCode", str(exit_code))
        except SystemExit as e:
            # argparse and explicit sys.exit() raise this. Honor the code.
            code = e.code
            if isinstance(code, int):
                exit_code = code
            elif code is None:
                exit_code = 0
            else:
                exit_code = 1
            if exit_code != 0:
                error_info = ("SystemExit", str(code))
        except KeyboardInterrupt:
            exit_code = 130
            error_info = ("KeyboardInterrupt", "interrupted by user")
            traceback.print_exc()
        except Exception as e:  # noqa: BLE001 - we genuinely want to catch all
            exit_code = 1
            error_info = (type(e).__name__, str(e))
            traceback.print_exc()
        finally:
            # If the script never registered a log path, register the
            # fallback so we always produce a file.
            if _log_handle is None:
                fb = default_log_path() if default_log_path else _fallback_log_path(script_name)
                try:
                    set_log_path(fb)
                except Exception:
                    pass

            elapsed = (_dt.datetime.utcnow() - started).total_seconds()
            elapsed_str = _format_elapsed(elapsed)
            log_str = str(_log_path) if _log_path else "(no log file written)"
            if error_info is None:
                term_line = (
                    f"\n[DONE] {script_name} completed in {elapsed_str} "
                    f"— log: {log_str}"
                )
            else:
                etype, emsg = error_info
                trail = " — see traceback above" if etype not in {"SystemExit", "ExitCode"} else ""
                term_line = (
                    f"\n[ERROR] {script_name} failed "
                    f"({etype}: {emsg}) after {elapsed_str} "
                    f"— log: {log_str}{trail}"
                )
            # Write to both the file (if open) and the real terminal.
            if _log_handle is not None:
                try:
                    _log_handle.write(term_line + "\n")
                    _log_handle.flush()
                except Exception:
                    pass
            try:
                if _orig_stdout is not None:
                    _orig_stdout.write(term_line + "\n")
                    _orig_stdout.flush()
            except Exception:
                pass

    return exit_code
