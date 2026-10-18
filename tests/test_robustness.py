from halo import robustness


def test_paired_statistics():
    r = robustness.paired([9.0, 8.0, 12.0], [10.0, 10.0, 10.0])
    assert r["wins"] == 2 and r["n"] == 3
    assert abs(r["diff"] - (-1 / 3)) < 1e-12
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_small_run_is_deterministic_and_complete():
    grids = {"static_gates": [{"threshold": 0.8}, {"threshold": 0.9}]}
    kw = dict(seeds=[1000, 1001], q_seeds=[0], grids=grids, workers=1, progress=lambda s: None)
    a = robustness.run(**kw)
    b = robustness.run(**kw)
    for cond in ("baseline", "stress"):
        ca, cb = a["conditions"][cond], b["conditions"][cond]
        assert [r["diff"] for r in ca["comparisons"]] == [r["diff"] for r in cb["comparisons"]]
        assert ca["best_strongest"] == "static_gates"
        names = [r["learner"] for r in ca["comparisons"]]
        assert len(names) == len(set(zip(names, (r["credit"] for r in ca["comparisons"]),
                                         (r["baseline"] for r in ca["comparisons"]))))
        assert {r["credit"] for r in ca["comparisons"]} == {"enforced", "proposed"}
    assert "Post hoc robustness" in robustness.format_report(a)
