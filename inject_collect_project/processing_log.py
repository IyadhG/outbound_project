"""
processing_log.py — Log entry factory module for the agentic pipeline.

Provides a single pure factory function with no class, no state, and no I/O.
"""

from datetime import datetime, timezone


def make_log_entry(gate: str, action: str, dqs_at_gate: float, **extra) -> dict:
    """
    Create a structured processing log entry.

    Args:
        gate:        Gate name string (e.g. "entity_validation", "data_quality",
                     "persona_worthiness").
        action:      Human-readable description of what was done at this gate.
        dqs_at_gate: DQS value at the time this gate was evaluated.
        **extra:     Additional gate-specific fields merged into the entry as
                     top-level keys.

    Returns:
        dict with base keys: gate, timestamp (ISO 8601 UTC), action, dqs_at_gate,
        plus any extra kwargs as additional top-level keys.
    """
    entry = {
        "gate": gate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "dqs_at_gate": dqs_at_gate,
    }
    entry.update(extra)
    return entry
