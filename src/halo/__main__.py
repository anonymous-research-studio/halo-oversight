"""
Command-line entry point.

    python -m halo all          # every table; writes results/
    python -m halo gate         # Table 2, randomized suite, mutation testing
    python -m halo allocation   # tuning, Tables 3 and 4, decision rule
    python -m halo bench        # adapter release latency with a stub agent
    python -m halo robustness   # post hoc robustness analysis (not preregistered)
    python -m halo intake FILE [FILE ...]   # run generated scenario files
"""

import argparse
import os
import sys
import time

from .experiments import allocation, gate_eval, report
from .verification import intake
from .verification.randomized import N_SEQUENCES


class Tee:
    """Print to stdout and collect the same text for summary.txt."""

    def __init__(self):
        self.parts = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.parts.append(text)

    def text(self):
        return "\n".join(self.parts) + "\n"


def run_gate(out, out_dir, n_sequences):
    t2 = gate_eval.table2()
    out(report.format_table2(t2))
    rand = gate_eval.randomized(n_sequences)
    out(report.format_randomized(rand))
    print("\n(running mutants against the randomized suite...)", flush=True)
    mut = gate_eval.mutation(n_sequences)
    out(report.format_mutation(mut))
    if out_dir:
        report.write_gate_csvs(out_dir, t2, rand, mut)


def run_allocation(out, out_dir):
    alloc = allocation.run(progress=lambda s: print(s, flush=True))
    out(f"\nQ-learner parameters: {alloc['q_params']}")
    out(report.format_tuning(alloc["tuning"]))
    out(report.format_table3(alloc["table3"]))
    out(report.format_table4(alloc["table4"]))
    out(report.format_decision(alloc["decision"],
                               "Decision rule vs. best protocol-specified non-learning policy"))
    out(report.format_decision(alloc["decision_incl_post_hoc"],
                               "Decision rule incl. post hoc comparator (for reference)"))
    holds = sum(r["gate_holds"] for rows in alloc["table3"].values() for r in rows.values())
    out(f"\nGate holds on simulated releases (should be 0): {holds}")
    if out_dir:
        report.write_allocation_csvs(out_dir, alloc)


def run_robustness(out, out_dir, workers):
    from .experiments import robustness
    res = robustness.run(workers=workers, progress=lambda s: print(s, flush=True))
    out(robustness.format_report(res))
    if out_dir:
        robustness.write_csvs(out_dir, res)


def run_bench(out, n=20000):
    from .experiments.bench import release_latency
    r = release_latency(n)
    out(report.header("Adapter release latency (deterministic stub agent)"))
    out(f"Median {r['median_us']:.1f} µs, 95th percentile {r['p95_us']:.1f} µs over {n:,} releases")
    out(f"Platform: {r['platform']}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m halo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["all", "gate", "allocation", "bench", "robustness", "intake"])
    parser.add_argument("files", nargs="*", help="scenario JSON files (intake only)")
    parser.add_argument("--out", default="results", help="output directory (default: results)")
    parser.add_argument("--sequences", type=int, default=N_SEQUENCES,
                        help=f"randomized sequences (default: {N_SEQUENCES})")
    parser.add_argument("--workers", type=int, default=None,
                        help="processes for the robustness analysis (default: all CPUs)")
    args = parser.parse_args(argv)

    if args.command == "intake":
        if not args.files:
            parser.error("intake needs at least one scenario file")
        rows = []
        for path in args.files:
            rows.extend(intake.evaluate(intake.load(path)))
        print(intake.format_report(rows))
        return 1 if any(r["valid"] and not r["agree"] for r in rows) else 0

    os.makedirs(args.out, exist_ok=True)
    out = Tee()
    start = time.time()
    out("HALO evaluation suite")
    if args.command in ("all", "gate"):
        run_gate(out, args.out, args.sequences)
    if args.command in ("all", "allocation"):
        run_allocation(out, args.out)
    if args.command in ("all", "bench"):
        run_bench(out)
    if args.command == "robustness":
        run_robustness(out, args.out, args.workers)
    out(f"\nCompleted in {time.time() - start:.0f} s")
    with open(os.path.join(args.out, f"summary_{args.command}.txt"), "w", encoding="utf-8") as f:
        f.write(out.text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
