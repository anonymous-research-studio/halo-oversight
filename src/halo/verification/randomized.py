"""
Randomized test suite: 20,000 event sequences from a fixed seed.

Each sequence draws:
- A registry of 1-4 reviewers
- A contract with up to 2 required roles and k in {1, 2}
- 12 events mixing reviews, revocations and release attempts
  with random times, outputs, revisions and actions

Every release attempt is compared in both directions with the oracle.
"""

import numpy as np

from ..gate import Contract, ReleaseGate, output_hash
from .oracle import Oracle, OracleContract


ROLES_POOL = ["general", "pharmacist", "specialist", "nurse", "attending"]
SEED = 12345
N_SEQUENCES = 20000
EVENTS_PER_SEQ = 12


def run_randomized_suite(
    gate_class=ReleaseGate,
    n_sequences: int = N_SEQUENCES,
    seed: int = SEED,
) -> dict:
    """Run randomized test suite against oracle.

    Returns dict with:
      - total_release_attempts
      - oracle_permits
      - disagreements: list of (seq_idx, event_idx, gate_result, oracle_result)
    """
    rng = np.random.RandomState(seed)

    total_attempts = 0
    oracle_permits = 0
    disagreements = []

    for seq_idx in range(n_sequences):
        gate = gate_class()
        oracle = Oracle()

        # Random registry: 1-4 reviewers
        n_reviewers = rng.randint(1, 5)
        reviewers = {}
        for r_idx in range(n_reviewers):
            rid = f"rev_{r_idx}"
            n_roles = rng.randint(1, 3)
            roles = set(rng.choice(ROLES_POOL, n_roles, replace=False))
            reviewers[rid] = roles
            gate.register_reviewer(rid, roles)
            oracle.register(rid, roles)

        # Random contract
        case_id = f"seq{seq_idx:05d}"
        output_text = f"output_{rng.randint(0, 1000000)}"
        h = output_hash(output_text)
        n_required = rng.randint(0, 3)
        required_roles = frozenset(rng.choice(ROLES_POOL, n_required, replace=False)) if n_required > 0 else frozenset()
        k = rng.choice([1, 2])
        floor = rng.randint(1, 8)
        t0 = rng.uniform(0, 50)
        deadline = t0 + rng.uniform(10, 100)

        contract = Contract(case_id, h, "v1", "o1", required_roles, k, floor, t0, deadline)
        gate.issue(contract)
        oracle.issue(OracleContract(case_id, h, "v1", "o1", required_roles, k, floor, t0, deadline))

        # Alternative outputs/revisions for mutation
        alt_outputs = [output_text, f"alt_{rng.randint(0, 1000000)}"]
        alt_rules = ["v1", "v2"]
        alt_obs = ["o1", "o2"]

        # 12 random events
        for ev_idx in range(EVENTS_PER_SEQ):
            event_type = rng.choice(["review", "revoke_reviewer", "release"], p=[0.4, 0.1, 0.5])

            if event_type == "review":
                # Random review
                rid = f"rev_{rng.randint(0, n_reviewers)}"
                decision = rng.choice(["approve", "reject"])
                time = rng.uniform(0, deadline + 20)
                gate.record_review(case_id, rid, decision, time)
                oracle.review(case_id, rid, decision, time)

            elif event_type == "revoke_reviewer":
                # Randomly revoke a reviewer
                rid = f"rev_{rng.randint(0, n_reviewers)}"
                gate.revoke_reviewer(rid)
                oracle.unregister(rid)

            elif event_type == "release":
                # Random release attempt
                total_attempts += 1
                out_text = rng.choice(alt_outputs)
                out_h = output_hash(out_text)
                rule_rev = rng.choice(alt_rules)
                obs_rev = rng.choice(alt_obs)
                t = rng.uniform(0, deadline + 20)
                action = rng.randint(1, 8)

                gate_permit, gate_reason = gate.release(
                    case_id, out_h, rule_rev, obs_rev, t, action
                )
                oracle_permit, oracle_reason = oracle.check_release(
                    case_id, out_h, rule_rev, obs_rev, t, action
                )

                if oracle_permit:
                    oracle_permits += 1

                if gate_permit != oracle_permit:
                    disagreements.append((
                        seq_idx, ev_idx,
                        (gate_permit, gate_reason),
                        (oracle_permit, oracle_reason),
                    ))

    return {
        "total_release_attempts": total_attempts,
        "oracle_permits": oracle_permits,
        "disagreements": disagreements,
    }


def run_mutant_detection(mutant_class, n_sequences=N_SEQUENCES, seed=SEED) -> bool:
    """Run randomized suite with a mutant gate. Returns True if mutant detected."""
    result = run_randomized_suite(
        gate_class=mutant_class,
        n_sequences=n_sequences,
        seed=seed,
    )
    return len(result["disagreements"]) > 0
