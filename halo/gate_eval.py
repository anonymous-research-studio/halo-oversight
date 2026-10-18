"""
Gate evaluation (Sections 2.2, 3.1, 3.2): scenario conformance (Table 2),
the randomized suite against the oracle, and mutation testing.
"""

from typing import Dict, List

from . import checker
from .mutants import MUTANTS
from .randomized_tests import N_SEQUENCES, run_mutant_detection, run_randomized_suite
from .scenarios import SCENARIOS, run_on_gate, scenario_cases


def table2() -> List[Dict]:
    """Scenario conformance of the gate, the static checker and the ordinal
    control over the 80 cases."""
    methods = {
        "HALO gate": lambda case_id, spec: run_on_gate(spec, case_id),
        "Static contract checker": lambda case_id, spec: checker.check(spec),
        "Ordinal-only control": lambda case_id, spec: checker.ordinal_only(spec),
    }
    cases = scenario_cases()
    rows = []
    for method, run in methods.items():
        matched = false_permits = false_holds = 0
        exp_holds = exp_permits = 0
        unencoded_total = unencoded_released = 0
        reason_mismatches = 0
        for case_id, spec in cases:
            permit, reason = run(case_id, spec)
            exp = spec["expected"]
            matched += permit == exp["permit"]
            if exp["permit"]:
                exp_permits += 1
                false_holds += not permit
            else:
                exp_holds += 1
                false_permits += permit
                if not permit and reason != exp["reason"]:
                    reason_mismatches += 1
            if spec["name"] == "unencoded_hazard":
                unencoded_total += 1
                unencoded_released += permit
        rows.append({
            "method": method, "matched": matched, "total": len(cases),
            "false_permits": false_permits, "expected_holds": exp_holds,
            "false_holds": false_holds, "expected_permits": exp_permits,
            "unencoded_released": unencoded_released, "unencoded_total": unencoded_total,
            "hold_reason_mismatches": reason_mismatches,
        })
    return rows


def randomized(n_sequences: int = N_SEQUENCES) -> Dict:
    result = run_randomized_suite(n_sequences=n_sequences)
    return {
        "sequences": n_sequences,
        "release_attempts": result["total_release_attempts"],
        "oracle_permits": result["oracle_permits"],
        "disagreements": len(result["disagreements"]),
    }


def _killed_by_scenarios(mutant_cls) -> bool:
    # Labels change no mechanism, so one label suffices.
    for idx, spec in enumerate(SCENARIOS):
        permit, _ = run_on_gate(spec, f"clinical_s{idx:02d}", gate=mutant_cls())
        if permit != spec["expected"]["permit"]:
            return True
    return False


def mutation(n_sequences: int = N_SEQUENCES) -> Dict:
    rows = []
    for name, cls, is_live in MUTANTS:
        by_scen = _killed_by_scenarios(cls)
        by_rand = run_mutant_detection(cls, n_sequences=n_sequences) if is_live else False
        rows.append({"mutant": name, "live": is_live,
                     "scenario": by_scen, "randomized": by_rand,
                     "either": by_scen or by_rand})
    live = [r for r in rows if r["live"]]
    return {
        "rows": rows,
        "live": len(live),
        "equivalent": len(rows) - len(live),
        "scenario_detects": sum(r["scenario"] for r in live),
        "randomized_detects": sum(r["randomized"] for r in live),
        "combined_detects": sum(r["either"] for r in live),
        "survive_both": [r["mutant"] for r in live if not r["either"]],
    }
