from halo.verification import checker
from halo.experiments.gate_eval import table2
from halo.verification.scenarios import SCENARIOS, run_on_gate


def test_checker_agrees_with_gate_on_every_scenario():
    for idx, spec in enumerate(SCENARIOS):
        assert checker.check(spec) == run_on_gate(spec, f"s{idx}"), spec["name"]


def test_table2_counts():
    rows = {r["method"]: r for r in table2()}
    for method in ("HALO gate", "Static contract checker"):
        r = rows[method]
        assert (r["matched"], r["false_permits"], r["false_holds"]) == (80, 0, 0)
        assert r["hold_reason_mismatches"] == 0
    ordinal = rows["Ordinal-only control"]
    assert (ordinal["matched"], ordinal["false_permits"], ordinal["false_holds"]) == (32, 48, 0)
    assert all(r["unencoded_released"] == 4 for r in rows.values())


def test_ordinal_control_only_checks_floor():
    spec = next(s for s in SCENARIOS if s["name"] == "changed_output")
    assert checker.ordinal_only(spec) == (True, None)
    spec = next(s for s in SCENARIOS if s["name"] == "action_below_floor")
    assert checker.ordinal_only(spec) == (False, "action_below_floor")
