"""Release latency of the agent adapter with a deterministic stub agent (Section 3.5)."""

import platform
import statistics
import time

from ..adapter import AgentAdapter


def stub_agent(prompt: str) -> str:
    return f"Draft note for: {prompt}"


def release_latency(n: int = 20000) -> dict:
    samples = []
    for i in range(n):
        adapter = AgentAdapter(stub_agent)
        adapter.register_reviewer("pharm1", {"pharmacist"})
        adapter.run(f"case {i}", required_roles=frozenset({"pharmacist"}),
                    min_reviewers=1, floor=5, deadline_seconds=1800)
        adapter.submit_review("pharm1", "approve")
        t = time.perf_counter()
        released = adapter.try_release(enforced_action=5)
        samples.append(time.perf_counter() - t)
        assert released is not None
    samples.sort()
    return {
        "median_us": statistics.median(samples) * 1e6,
        "p95_us": samples[int(0.95 * (len(samples) - 1))] * 1e6,
        "platform": f"{platform.python_implementation()} {platform.python_version()}, "
                    f"{platform.machine()}",
    }
