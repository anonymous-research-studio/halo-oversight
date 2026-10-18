# HALO: obligation-preserving release gate

This repository contains the code and results for the paper *Flexible Allocation, Fixed Obligations: Verifying Human Oversight in Compound AI Systems*. HALO separates two questions that oversight systems tend to merge:

- **Allocation:** how much oversight a case should get. A rule-based or learned policy proposes one of seven modalities.
- **Authorization:** whether the oversight the rules require actually happened for the output being released. A deterministic gate checks this.

The gate permits a release only if every required role approved *this exact output* under the current rule and observation revisions, inside the release window, with no outstanding qualified rejection and at least *k* distinct qualified approvers.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"        # or: pip install -r requirements.txt
python -m halo all              # every table in the paper, about 2 minutes
pytest                          # 31 tests, about 2 minutes
```

`python -m halo all` prints the tables and writes them to `results/`. `run_all.py` does the same thing.

Requirements are Python 3.9 or later and NumPy 1.21 or later. Tested on Python 3.9.6 with NumPy 1.21.6 and on Python 3.13.7 with NumPy 2.5.4; both give byte-identical result files.

## Commands

| Command | What it produces |
|---|---|
| `python -m halo gate` | Table 2 (scenario conformance), the randomized suite against the oracle, and mutation testing |
| `python -m halo allocation` | Tuning choices, Table 3 (matched-policy comparison), Table 4 (rule coverage) and the decision-rule check |
| `python -m halo bench` | Adapter release latency with a deterministic stub agent |
| `python -m halo robustness` | Post hoc robustness analysis (not preregistered), about 4 minutes on 8 cores |
| `python -m halo all` | All of the above |
| `python -m halo intake FILE...` | Runs generated scenario files through the gate, oracle and static checker |

Options: `--out DIR` (default `results/`), `--sequences N` for the randomized suite (default 20,000) and `--workers N` for the robustness analysis (default: all CPUs).

## Repository layout

```
halo/
  gate.py              Release gate: issue, record_review, revoke, release (Section 2.1)
  oracle.py            Separately written oracle for the release condition
  checker.py           Static contract checker and ordinal-only control (Table 2)
  scenarios.py         The 20 workflow scenarios, as data
  randomized_tests.py  20,000 randomized event sequences compared with the oracle
  mutants.py           16 single-check mutants of the release method
  gate_eval.py         Table 2, randomized suite, mutation testing
  simulator.py         Discrete-event simulator for the allocation study (Section 2.3)
  policies.py          The eight allocation policies and their tuning grids
  allocation.py        Tuning, training, evaluation, Tables 3 and 4, decision rule
  robustness.py        Post hoc robustness analysis
  adapter.py           Places the gate in front of any generate(prompt) callable
  intake.py            Runs externally generated scenario files (Section 2.4)
  bench.py             Release-latency benchmark
  report.py            Text tables and CSV output
examples/generated_scenarios.json   Example input for the intake
results/                            Output of `python -m halo all`
tests/                              pytest suite
```

## Output files

| File | Contents |
|---|---|
| `summary_all.txt` | Everything printed by the run |
| `table2_conformance.csv` | Matched decisions, false permits and false holds for the gate, checker and control |
| `randomized.csv` | Release attempts, oracle permits and disagreements |
| `mutation.csv` | Which suite detects each mutant |
| `tuning.csv` | Validation cost of every configuration tried, and the one chosen |
| `table3_policies.csv` | Table 3 for both conditions, including the cost on each test seed |
| `table4_coverage.csv` | Harm at coverage 1.0, 0.5 and 0.0, and unencoded hazards released |
| `decision_rule.csv` | The decision rule applied to each learner |
| `summary_robustness.txt` | Everything printed by the robustness analysis |
| `robustness_policies.csv` | Each policy on the 50 fresh seeds, and the strongest baselines |
| `robustness_comparisons.csv` | Paired cost differences with 95% confidence intervals and wins |

All results are deterministic except the latency benchmark, which depends on the machine.

## How the evaluation works

### Gate (Sections 2.2, 3.1, 3.2)

- **Scenarios:** 20 scenarios run under four namespace labels, giving 80 cases (28 expected permits, 52 expected holds). The labels change no mechanism.
- **Three evaluators per scenario:** the gate replays the events; the static checker evaluates the release condition directly from the scenario fields, sharing no code with the gate; the ordinal-only control checks only the floor of Equation 1.
- **Randomized suite:** 20,000 sequences from a fixed seed. Each draws one to four reviewers, a contract with up to two required roles and *k* in {1, 2}, and twelve events. Every release attempt is compared with the oracle.
- **Mutants:** 16 mutants each remove or weaken one check. Two are equivalent under the current interface, because receipts copy the hash and revisions from the stored contract.

### Allocation study (Sections 2.3, 3.3 to 3.5)

1. **Workload.** All randomness for a run (arrivals, cases, rule firings, catch draws) is drawn before any policy acts. Every policy therefore sees identical cases on a given seed, and differences between policies come only from their decisions.
2. **Tuning.** Each tunable policy tries at most eight configurations and keeps the one with the lowest mean cost on validation seeds 100 to 102 (baseline condition, full rule coverage).
3. **Training.** The LinUCB bandit and the linear Q-learner are trained on seeds 200 to 202 in sequence, then frozen: no further updates and no exploration. Their configurations are tuned the same way.
4. **Testing.** Each policy is evaluated on seeds 300 to 304. The capacity-stress condition (1.6 arrivals per minute, 2 generalists and 1 specialist) is never used for tuning or training.
5. **Decision rule.** A learner counts as better than the best protocol-specified non-learning policy only if its mean cost is more than 5% lower *and* its cost is lower on every test seed. The two-threshold policy was added post hoc and is excluded from this comparison; the output also shows the comparison with it included.
6. **Gate in the loop.** Every simulated release goes through a `ReleaseGate`. Obligated cases need a specialist approval receipt, and the run reports the number of gate holds (always 0).

**Rule coverage.** Coverage *c* means each of the three rules is applied to a case with probability *c*, independently for each case and rule. Rules fire on observable features, never on hazard labels. One hazard type (`outdated_guideline`) has no rule at any coverage level.

### Post hoc robustness analysis

`python -m halo robustness` was added after the protocol results were seen and is not part of the preregistered protocol. It reuses the protocol's tuning and training, then evaluates on 50 fresh seeds (1000 to 1049) that were never used for tuning or training. It covers five checks:

- **Fresh seeds:** every protocol-tuned policy on the 50 new seeds, with paired cost differences, 95% confidence intervals and win counts.
- **Strongest baseline:** each rule-based policy tuned on the 50 test seeds themselves over a finer grid. This deliberately favors the baselines, so a learner that still wins is not beating an under-tuned comparator.
- **Q-learner seeds:** the tuned Q-learner retrained with exploration seeds 0 to 4.
- **Credit assignment:** learners credited with the action they proposed instead of the action enforced after the rule floor, then retuned and retrained.
- **Breakeven effort weight:** the reviewer-time weight at which the bandit and the strongest baseline cost the same.

**Cost.** Each run is scored as harm plus 0.02 per reviewer minute plus 1.0 per withheld case, per 1,000 arrivals. The weights are assumptions, so the components are reported separately.

## Generated scenarios

The intake accepts JSON files using the schema documented at the top of `halo/scenarios.py`. The `expected` field is optional. The intake only reports results: it never changes rules, contracts or the gate.

```bash
python -m halo intake examples/generated_scenarios.json
```

The command exits with status 1 if the gate, oracle and checker disagree on any valid scenario.

## What the gate trusts and does not check

The gate trusts the contracts, receipts, reviewer registry, revision identifiers and clock it is given, and it assumes a single gate instance. It does not check what the reviewer's screen displayed, and distinct reviewer identities do not guarantee independent judgments. It does not stop the originator of an action from approving it. Authorizations live in memory, so replay protection does not survive a restart.

The oracle and the static checker were written by the same team from the same specification as the gate, so a misunderstanding common to all three would pass unnoticed.

## Changes from the earlier version

See [CHANGES.md](CHANGES.md).

## License

MIT. See [LICENSE](LICENSE).
