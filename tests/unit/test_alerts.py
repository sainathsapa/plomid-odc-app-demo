"""Alert manager tests — rules, cooldowns, timers. No model, no clock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odc.alerts import AlertManager, WebhookSender

T0 = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)


def _row(seat_id: str, **kw) -> dict:
    row = {
        "seat_id": seat_id,
        "presence": "UNKNOWN",
        "child": "Unknown",
        "restraint": "Unknown",
        "belt": "Unknown",
        "belt_issue": False,
        "dms": None,
        "reason_codes": [],
    }
    row.update(kw)
    return row


def test_belt_issue_raises_cools_down_and_clears():
    manager = AlertManager(cooldown_s=60.0)
    rows = [_row("driver", presence="OCCUPIED", belt="Not detected", belt_issue=True)]
    active, raised, cleared = manager.update(rows, "ses-1", 10, T0)
    assert len(active) == 1 and len(raised) == 1 and cleared == []
    assert raised[0]["severity"] == "WARNING" and raised[0]["kind"] == "belt"
    # Still active: no duplicate raise.
    active, raised, _ = manager.update(rows, "ses-1", 11, T0 + timedelta(seconds=5))
    assert len(active) == 1 and raised == []
    # Resolved -> cleared; re-raise inside cooldown suppressed.
    ok = [_row("driver", presence="OCCUPIED", belt="Detected")]
    _, _, cleared = manager.update(ok, "ses-1", 12, T0 + timedelta(seconds=10))
    assert len(cleared) == 1
    active, raised, _ = manager.update(rows, "ses-1", 13, T0 + timedelta(seconds=30))
    assert raised == [] and active == []
    # After cooldown the same condition raises again.
    active, raised, _ = manager.update(rows, "ses-1", 14, T0 + timedelta(seconds=100))
    assert len(raised) == 1


def test_dms_flags_raise_preview_alerts():
    manager = AlertManager()
    rows = [
        _row("driver", presence="OCCUPIED", dms={"drowsy": True, "distracted": False,
                                                "phone": False, "yawning": False})
    ]
    active, raised, _ = manager.update(rows, "ses-1", 3, T0)
    assert [a["kind"] for a in raised] == ["dms-drowsy"]
    assert "preview" in raised[0]["message"]


def test_unattended_timer_needs_duration_and_resets():
    manager = AlertManager(unattended_after_s=600.0)
    rows = [_row("rear", presence="OCCUPIED")]
    active, raised, _ = manager.update(rows, "ses-1", 1, T0)
    assert raised == []
    active, raised, _ = manager.update(rows, "ses-1", 2, T0 + timedelta(seconds=601))
    assert len(raised) == 1 and raised[0]["kind"] == "unattended"
    assert "not connected" in raised[0]["message"]
    # Seat empties -> timer resets, alert clears.
    _, _, cleared = manager.update([_row("rear", presence="EMPTY_CANDIDATE")], "ses-1", 3, T0)
    assert len(cleared) == 1
    active, raised, _ = manager.update(rows, "ses-1", 4, T0 + timedelta(seconds=602))
    assert raised == []  # timer restarted, not enough held time


def test_webhook_sender_never_blocks():
    sender = WebhookSender("http://127.0.0.1:9/nope", timeout_s=0.2)
    sender.start()
    for i in range(25):  # over capacity: oldest dropped, never raises
        sender.send({"i": i})
    assert sender.dropped >= 1
