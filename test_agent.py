from src.agent import decide


def test_sensitive_request_escalates():
    auto, reason = decide("Please give me my password", 0.95, [("example", 0.9)])
    assert auto is False
    assert "Sensitive" in reason


def test_low_confidence_escalates():
    auto, reason = decide("something weird happened", 0.55, [("example", 0.9)])
    assert auto is False


def test_weak_evidence_escalates():
    auto, reason = decide("app issue", 0.95, [("example", 0.1)])
    assert auto is False


def test_confident_and_grounded_auto_handles():
    auto, reason = decide("my app is slow after update", 0.95, [("app is slow after update", 0.9)])
    assert auto is True
