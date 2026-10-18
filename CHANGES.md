# Changes

## 1.1.0

This version fixes defects in the earlier, unpublished version of the code (1.0) and brings the code in line with the methods the paper describes. The gate (`halo/gate.py`), the oracle and the 16 mutants are unchanged, and so are the gate results: Table 2, the randomized suite and the mutation results reproduce exactly. The allocation results (Tables 3 and 4) change.

### Gate evaluation

- **Static contract checker.** In 1.0 the checker's row in Table 2 reprinted the gate's numbers. It is now a separate evaluator (`halo/checker.py`) that works from the scenario fields and shares no code with the gate.
- **Ordinal-only control.** In 1.0 the control's results were hard-coded by scenario name. It is now computed by checking the floor of Equation 1.
- **Scenarios as data.** The 20 scenarios are now data, so the gate, the checker, the control and the intake all run the same definitions. The scenario named `future_dated_review` in 1.0 is renamed `pre_window_review`, because its review is dated before the window opens. Its behavior is unchanged.
- **Combined detection.** Mutation testing now reports detection by either suite: 13 of 14 live mutants, with only `M15_WeakenWindowExclDeadline` surviving both.

### Allocation study

- **Matched workloads.** In 1.0, hazard-catch draws shared a random stream with case generation, so a policy's own decisions changed which cases arrived later. On the same seed, different policies saw different numbers of hazardous and obligated cases. All randomness for a run is now drawn before any policy acts.
- **Rule 3.** It compared `component_count / 8` with 3 and could never fire, so only two rules were active in 1.0. It now compares with 3/8.
- **Tuning.** In 1.0 the validation seeds and tuning grids were defined but never used, and every threshold was a fixed default. Each tunable policy is now tuned on the validation seeds.
- **Training.** In 1.0, Table 3 created a fresh learner for each training seed, so learners were effectively trained on one seed, while Table 4 trained on all three. Both tables now use one learner trained on all three seeds.
- **Freezing.** In 1.0, both learners kept updating during testing and in the stress condition; "freezing" only set the Q-learner's exploration rate to 0. `freeze()` now stops all updates.
- **Q-learner reproducibility.** In 1.0, exploration used NumPy's unseeded global generator, so results changed from run to run. It now uses its own seeded generator.
- **Q-learner TD target.** In 1.0 it bootstrapped on the same case's features. It now bootstraps on the next arriving case.
- **Discretionary precision.** In 1.0, missed hazards under blocking review counted as true positives, but missed hazards under monitoring were left out entirely. Every discretionary intervention on a hazardous case is now a true positive, and every one on a safe case a false positive.
- **Table 4's unencoded-hazard column.** In 1.0 it measured all uncaught hazards. It now measures released hazards of the type no rule encodes (`outdated_guideline`).
- **Decision rule.** The per-seed sign condition was never checked in 1.0. It is now applied and reported.
- **Standard deviation.** Cost is now reported with the sample standard deviation (1.0 used the population standard deviation).
- **Gate in the loop.** The simulator imported the gate but never used it. Every simulated release now passes through it.

### New

- `halo/intake.py`: runs generated scenario files through the gate, the oracle and the checker.
- `halo/bench.py`: adapter release-latency benchmark.
- `halo/robustness.py`: post hoc robustness analysis (50 fresh seeds, strongest baselines, Q-learner seed sensitivity, credit assignment, breakeven effort weight). It is labeled as not preregistered.
- A command-line interface (`python -m halo`), CSV output, a pytest suite and packaging metadata.
