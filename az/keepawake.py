"""Keep Windows awake during long training runs.

Windows' idle-sleep timer is driven by *user input* and explicit power
requests — NOT by CPU load. So a machine pegged at 100% for an hour with no
mouse/keyboard activity still sleeps (confirmed in the event log). Calling
SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) asserts a
"system required" request that holds for the lifetime of the process,
overriding the idle timer regardless of the power-scheme timeout. No admin
needed. (Does NOT survive the controlling app being closed — that's a
separate concern handled by detaching the run.)
"""
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake():
    """Best-effort: prevent idle sleep while this process lives."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
    except Exception:
        pass
