# HALO: Human-AI Layered Oversight

[![tests](https://github.com/anonymous-research-studio/halo-oversight/actions/workflows/tests.yml/badge.svg)](https://github.com/anonymous-research-studio/halo-oversight/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and results for the paper **Flexible Allocation, Fixed Obligations: Verifying Human Oversight in Compound AI Systems**.

Requiring human review of an AI output does not show that the right reviewer approved the output that was finally released. HALO separates two decisions:

- **Allocation** (flexible): how much oversight a case gets. A rule-based or learned policy proposes one of seven intervention modalities, from passive telemetry to human takeover.
- **Authorization** (fixed): whether the required oversight actually happened. A deterministic release gate permits a release only if every required role approved *this exact output*, under the current rule and observation revisions, inside the release window, with no outstanding qualified rejection and at least *k* distinct qualified approvers.

## Installation

Requires Python 3.9 or later and NumPy 1.21 or later.

```bash
git clone https://github.com/anonymous-research-studio/halo-oversight.git
cd halo-oversight
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

## Using the gate

```python
from halo import Contract, ReleaseGate, output_hash

gate = ReleaseGate()
gate.register_reviewer("pharm1", {"pharmacist"})

note = "Increase metformin to 1000 mg twice daily"
gate.issue(Contract(
    case_id="case-1", output_hash=output_hash(note),
    rule_revision="v1", obs_revision="o1",
    required_roles=frozenset({"pharmacist"}), min_reviewers=1,
    min_action=5, window_start=0.0, deadline=60.0,
))
gate.record_review("case-1", "pharm1", "approve", timestamp=10.0)

# The model regenerates the note after a new lab result arrives.
gate.release("case-1", output_hash("Increase metformin to 1500 mg twice daily"), "v1", "o1", 20.0, 5)
# -> (False, HoldReason.payload_changed)

gate.release("case-1", output_hash(note), "v1", "o1", 20.0, 5)
# -> (True, None)
```

`halo.AgentAdapter` wraps any `generate(prompt)` callable with the same gate, voiding approvals whenever the output is regenerated.

## Reproducing the paper

Every result is deterministic, except the latency benchmark, which depends on the machine.

| Paper | Command | Output |
|---|---|---|
| Table 2, Sections 3.1 and 3.2 | `halo gate` | `results/gate/` |
| Tables 3 and 4, Sections 3.3 to 3.5 | `halo allocation` | `results/allocation/` |
| Latency, Section 3.5 | `halo bench` | `results/summary_all.txt` |
| All of the above (about 2 minutes) | `halo all` | `results/` |
| Table 5, Section 3.6 (about 4 minutes on 8 cores) | `halo robustness` | `results/robustness/` |

`halo` is installed as a command; `python -m halo` works the same way. Options: `--out DIR` (default `results/`), `--sequences N` for the randomized suite (default 20,000) and `--workers N` for the robustness analysis.

| File | Contents |
|---|---|
| `results/gate/table2_conformance.csv` | Decisions of the gate, static checker and ordinal control on the 80 scenario cases |
| `results/gate/randomized.csv` | Release attempts, oracle permits and disagreements over 20,000 sequences |
| `results/gate/mutation.csv` | Which suite detects each of the 16 mutants |
| `results/allocation/tuning.csv` | Validation cost of every configuration tried, and the one chosen |
| `results/allocation/table3_policies.csv` | Table 3, both conditions, with per-seed costs |
| `results/allocation/table4_coverage.csv` | Table 4: harm by rule coverage, unencoded hazards released |
| `results/allocation/decision_rule.csv` | The preregistered decision rule applied to each learner |
| `results/robustness/comparisons.csv` | Table 5: paired cost differences on 50 fresh seeds, with 95% intervals |
| `results/robustness/policies.csv` | Every policy on the 50 fresh seeds, and the strongest tuned baselines |
| `results/summary_all.txt`, `results/summary_robustness.txt` | Everything the runs print |

## Repository structure

```
src/halo/
  gate.py               release gate: issue, record_review, revoke, release
  adapter.py            places the gate in front of a generate(prompt) callable
  verification/
    oracle.py           separately written oracle for the release condition
    checker.py          static contract checker and ordinal-only control
    scenarios.py        the 20 workflow scenarios, as data
    randomized.py       randomized event sequences compared with the oracle
    mutants.py          16 single-check mutants of the release method
    intake.py           runs externally generated scenario files
  simulation/
    simulator.py        discrete-event simulator of review queues
    policies.py         the eight allocation policies and their tuning grids
  experiments/
    gate_eval.py        Table 2, randomized suite, mutation testing
    allocation.py       tuning, training, Tables 3 and 4, decision rule
    robustness.py       post hoc analysis behind Table 5
    bench.py            release-latency benchmark
    report.py           text tables and CSV output
tests/                  pytest suite
examples/               example generated-scenario file for the intake
results/                output of the commands above
```

## Evaluation protocol

**Gate.** Twenty workflow scenarios, each run under four namespace labels, give 80 cases with 28 expected permits and 52 expected holds. Three evaluators run the same scenario data: the gate, a static checker that shares no code with it, and an ordinal control that checks only the oversight floor. A randomized suite of 20,000 event sequences compares every release attempt with the oracle, and 16 mutants (14 behavior-changing) test whether either suite would notice a faulty gate.

**Allocation.** All randomness for a simulated run is drawn before any policy acts, so every policy faces identical cases on a given seed, and every release passes through the gate. Tunable policies choose among at most eight configurations on validation seeds 100 to 102. The two learners are trained on seeds 200 to 202 and then frozen. Everything is evaluated on test seeds 300 to 304. A learner counts as better than the best protocol-specified non-learning policy only if its mean cost is more than 5% lower and its cost is lower on every test seed. The capacity-stress condition is never used for tuning or training.

**Robustness (post hoc, not preregistered).** The same frozen policies are evaluated on 50 fresh seeds (1000 to 1049). The analysis also retunes the rule-based baselines on those seeds, retrains the Q-learner with five exploration seeds, and credits learners with their proposed rather than enforced action.

## Generated scenarios

```bash
halo intake examples/generated_scenarios.json
```

The intake runs each scenario through the gate, the oracle and the static checker, and exits with status 1 if they disagree. It only reports results: it never changes rules, contracts or the gate. The schema is documented in `src/halo/verification/scenarios.py`.

## Testing

```bash
pytest          # 31 tests, about 2 minutes
```

GitHub Actions runs the suite on Python 3.9 and 3.12 for every push.

## Scope and limitations

The gate trusts the contracts, receipts, reviewer registry, revision identifiers and clock it is given, and assumes a single gate instance. It does not check what a reviewer's screen displayed, does not treat distinct reviewer identities as independent judgments, and does not stop the originator of an action from approving it. Authorizations live in memory, so replay protection does not survive a restart. The oracle and the static checker were written from the same specification as the gate, so a misunderstanding common to all three would pass unnoticed. The simulator's parameters are assumptions; only comparisons between policies are meaningful.

## Citation

If you use this code, please cite the paper. Citation metadata is in [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button reads it.

## License

MIT. See [LICENSE](LICENSE).
