import logging
import os
from pathlib import Path
from logging.handlers import MemoryHandler

# Disable logging's internal exception printing globally to avoid
# "Logged from file ..." crashes/spam when an underlying stream/file
# raises OSError (e.g., broken pipe, detached TTY, ephemeral CI handles).
# This does not change log levels or mute logs; it only prevents logging
# from propagating handler I/O errors to stderr.
logging.raiseExceptions = False

_LOGGER_CACHE: dict[str, logging.Logger] = {}
_MEMORY_HANDLERS: list[MemoryHandler] = []


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that swallows transient I/O errors.

    - Never raises on emit; avoids Windows/CI broken pipe issues (OSError: [Errno 5]).
    - Keeps formatting and levels identical to base class.
    - No environment gating; always safe.
    """

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        # Swallow all handler-level errors silently to keep hot paths safe.
        # Base implementation prints diagnostics when logging.raiseExceptions is True;
        # we explicitly avoid that to prevent test noise and crashes.
        return

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            super().emit(record)
        except (OSError, BrokenPipeError, ValueError):
            # Ignore failed writes (e.g., closed streams). Do not re-raise.
            pass


class _SafeFileHandler(logging.FileHandler):
    """FileHandler that tolerates transient file I/O errors.

    Uses the same interface as FileHandler but never raises from emit/handleError.
    This prevents save/log failures from destabilizing long-running jobs.
    """

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        return

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            super().emit(record)
        except OSError:
            # Drop the record if the filesystem is unavailable.
            pass


def get_logger(name: str = "omnicoder") -> logging.Logger:
    global _LOGGER_CACHE
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    # Ensure the root logger has at least one safe handler so plain `logging.*`
    # calls in other modules cannot crash on I/O errors.
    _root = logging.getLogger()
    if not _root.handlers:
        _root.setLevel(logging.INFO)
        _root.addHandler(_SafeStreamHandler())

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured elsewhere
        _LOGGER_CACHE[name] = logger
        return logger

    # Always-on verbose logging (no env gating). Use DEBUG for all omnicoder.* loggers.
    level = logging.DEBUG if name.startswith("omnicoder") else logging.INFO
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # Console handler (safe)
    sh = _SafeStreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    # Always enable buffered file logging for hot loggers to amortize IO without
    # reducing verbosity. No environment toggles.
    use_buffer = bool(name.startswith("omnicoder.model") or name.startswith("omnicoder.att") or name.startswith("omnicoder.moe") or name.startswith("omnicoder"))

    # Optional file handler (enabled unless explicitly disabled)
    log_path = os.getenv("OMNICODER_LOG_FILE", "tests_logs/omnicoder.log").strip()
    if log_path:
        try:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = _SafeFileHandler(p, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            if use_buffer:
                # Buffer INFO/DEBUG logs and flush rarely to reduce per-record IO.
                # Fixed capacity (no env gating); flush on CRITICAL and at shutdown.
                buf_cap = 65536
                mh = MemoryHandler(capacity=buf_cap, flushLevel=logging.CRITICAL, target=fh)
                logger.addHandler(mh)
                try:
                    _MEMORY_HANDLERS.append(mh)
                except Exception:
                    pass
            else:
                logger.addHandler(fh)
        except Exception:
            logging.getLogger('omnicoder.log').warning('file handler setup failed', exc_info=True)

    # Attach console handler last; keep console at INFO to avoid overwhelming TTY,
    # while file handlers capture full DEBUG. No env gating.
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.propagate = False
    _LOGGER_CACHE[name] = logger
    return logger


def flush_all_log_buffers() -> None:
    """
    Flush all MemoryHandler buffers created by get_logger.

    This does not reduce logging verbosity; it merely forces pending buffered
    records to be written to their underlying FileHandlers. Safe to call at the
    end of long-running benchmarks or training stages to ensure logs are on disk.
    """
    try:
        for mh in list(_MEMORY_HANDLERS):
            try:
                mh.flush()
            except Exception:
                # best-effort flush; continue others
                continue
    except Exception:
        logging.getLogger('omnicoder.log').warning('flush_all_log_buffers failed', exc_info=True)


