"""Shared helpers. Importing this package makes console output UTF-8 safe (Windows Git Bash defaults to cp1252)."""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - not a real text stream (e.g. under pytest capture)
        pass
