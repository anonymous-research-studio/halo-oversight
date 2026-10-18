import os

from halo.verification import intake
from halo.adapter import AgentAdapter
from halo.gate import HoldReason

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "generated_scenarios.json")


def _adapter(outputs):
    it = iter(outputs)
    a = AgentAdapter(lambda prompt: next(it))
    a.register_reviewer("pharm1", {"pharmacist"})
    return a


def test_approved_output_is_released_once():
    a = _adapter(["dose 10 mg"])
    a.run("p", required_roles=frozenset({"pharmacist"}), floor=5)
    assert a.try_release() is None
    a.submit_review("pharm1", "approve")
    assert a.try_release() == "dose 10 mg"
    assert a.try_release() is None


def test_regeneration_voids_earlier_approval():
    a = _adapter(["dose 10 mg", "dose 20 mg"])
    a.run("p", required_roles=frozenset({"pharmacist"}), floor=5)
    a.submit_review("pharm1", "approve")
    a.regenerate("p")
    assert a.try_release() is None
    assert a.last_hold_reason == HoldReason.missing_role_coverage
    a.submit_review("pharm1", "approve")
    assert a.try_release() == "dose 20 mg"


def test_rule_revision_change_holds():
    a = _adapter(["dose 10 mg"])
    a.run("p", required_roles=frozenset({"pharmacist"}), floor=5)
    a.submit_review("pharm1", "approve")
    a.set_rule_revision("v2")
    assert a.try_release() is None
    assert a.last_hold_reason == HoldReason.rules_changed


def test_example_scenarios_run_and_agree():
    rows = intake.evaluate(intake.load(EXAMPLE))
    assert len(rows) == 4
    assert all(r["valid"] and r["agree"] and r["matches_expected"] for r in rows)


def test_invalid_scenario_is_reported_not_run():
    rows = intake.evaluate([{"name": "bad", "events": [{"type": "explode"}]}])
    assert not rows[0]["valid"]
