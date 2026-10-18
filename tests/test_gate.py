from halo.gate import Contract, HoldReason, ReleaseGate, output_hash
from halo.scenarios import SCENARIOS, run_all_scenarios


def _two_role_gate(k):
    g = ReleaseGate()
    h = output_hash("note")
    g.register_reviewer("rev1", {"general", "pharmacist"})
    g.issue(Contract("c", h, "v1", "o1", frozenset({"general", "pharmacist"}), k, 5, 0.0, 100.0))
    return g, h


def test_all_80_scenario_cases_match():
    rows = run_all_scenarios()
    assert len(rows) == 80
    assert sum(r["expected_permit"] for r in rows) == 28
    for r in rows:
        assert r["permit"] == r["expected_permit"], r["case_id"]
        if not r["expected_permit"]:
            assert r["reason"] == r["expected_reason"], r["case_id"]


def test_scenario_count():
    assert len(SCENARIOS) == 20


def test_single_reviewer_both_roles_k1_permits():
    g, h = _two_role_gate(k=1)
    g.record_review("c", "rev1", "approve", 10.0)
    assert g.release("c", h, "v1", "o1", 50.0, 5) == (True, None)


def test_single_reviewer_both_roles_k2_holds_even_with_repeated_receipts():
    g, h = _two_role_gate(k=2)
    g.record_review("c", "rev1", "approve", 10.0)
    g.record_review("c", "rev1", "approve", 11.0)
    assert g.release("c", h, "v1", "o1", 50.0, 5) == (False, HoldReason.too_few_distinct_reviewers)


def test_approval_without_required_role_does_not_count_toward_k():
    g, h = _two_role_gate(k=2)
    g.register_reviewer("nurse1", {"nurse"})
    g.record_review("c", "rev1", "approve", 10.0)
    g.record_review("c", "nurse1", "approve", 12.0)
    assert g.release("c", h, "v1", "o1", 50.0, 5) == (False, HoldReason.too_few_distinct_reviewers)


def test_later_approval_does_not_override_qualified_rejection():
    g = ReleaseGate()
    h = output_hash("note")
    g.register_reviewer("p1", {"pharmacist"})
    g.register_reviewer("p2", {"pharmacist"})
    g.issue(Contract("c", h, "v1", "o1", frozenset({"pharmacist"}), 1, 5, 0.0, 100.0))
    g.record_review("c", "p1", "reject", 10.0)
    g.record_review("c", "p2", "approve", 20.0)
    assert g.release("c", h, "v1", "o1", 50.0, 5) == (False, HoldReason.qualified_rejection)


def test_release_at_deadline_instant_is_permitted():
    g = ReleaseGate()
    h = output_hash("note")
    g.register_reviewer("r", {"general"})
    g.issue(Contract("c", h, "v1", "o1", frozenset({"general"}), 1, 5, 0.0, 50.0))
    g.record_review("c", "r", "approve", 50.0)
    assert g.release("c", h, "v1", "o1", 50.0, 5) == (True, None)


def test_record_review_takes_roles_from_registry():
    g, _ = _two_role_gate(k=1)
    receipt = g.record_review("c", "rev1", "approve", 10.0)
    assert receipt.roles_at_review == frozenset({"general", "pharmacist"})
    assert g.record_review("missing", "rev1", "approve", 10.0) is None
