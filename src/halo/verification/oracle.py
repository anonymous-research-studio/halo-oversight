"""
Independent oracle for the release condition.

Evaluates the release condition from an event log without importing
anything from gate.py. Written from the same specification by the same
team; a misunderstanding common to both would pass unnoticed.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import hashlib


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class OracleContract:
    case_id: str
    output_hash: str
    rule_rev: str
    obs_rev: str
    roles: FrozenSet[str]
    min_k: int
    floor: int
    t0: float
    deadline: float


@dataclass
class OracleReceipt:
    case_id: str
    reviewer: str
    roles_at_review: FrozenSet[str]
    output_hash: str
    rule_rev: str
    obs_rev: str
    decision: str
    time: float


class Oracle:
    """Evaluates release decisions from an event log.

    Maintains the same state as the gate but in a separate implementation.
    """

    def __init__(self):
        self._contracts: Dict[str, OracleContract] = {}
        self._receipts: Dict[str, List[OracleReceipt]] = {}
        self._registry: Dict[str, Set[str]] = {}
        self._released: Set[str] = set()

    def register(self, reviewer_id: str, roles: Set[str]):
        self._registry[reviewer_id] = set(roles)

    def unregister(self, reviewer_id: str):
        self._registry.pop(reviewer_id, None)

    def issue(self, c: OracleContract):
        self._contracts[c.case_id] = c
        self._receipts.setdefault(c.case_id, [])

    def review(self, case_id: str, reviewer: str, decision: str, time: float):
        if case_id not in self._contracts:
            return
        c = self._contracts[case_id]
        roles = frozenset(self._registry.get(reviewer, set()))
        self._receipts[case_id].append(OracleReceipt(
            case_id=case_id,
            reviewer=reviewer,
            roles_at_review=roles,
            output_hash=c.output_hash,
            rule_rev=c.rule_rev,
            obs_rev=c.obs_rev,
            decision=decision,
            time=time,
        ))

    def revoke_case(self, case_id: str):
        self._contracts.pop(case_id, None)
        self._receipts.pop(case_id, None)

    def check_release(
        self,
        case_id: str,
        out_hash: str,
        rule_rev: str,
        obs_rev: str,
        t: float,
        action: int,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate release condition. Returns (permitted, reason_if_held)."""
        # Known case
        if case_id not in self._contracts:
            return False, "unknown_case"
        c = self._contracts[case_id]

        # Not already released
        if case_id in self._released:
            return False, "already_released"

        # Hash match
        if out_hash != c.output_hash:
            return False, "payload_changed"

        # Revision matches
        if rule_rev != c.rule_rev:
            return False, "rules_changed"
        if obs_rev != c.obs_rev:
            return False, "observations_changed"

        # Release window
        if t < c.t0 or t > c.deadline:
            return False, "outside_release_window"

        # Action floor
        if action < c.floor:
            return False, "action_below_floor"

        # Role-based checks
        if c.roles:
            cutoff = min(t, c.deadline)
            # Gather applicable receipts (hash, revisions, time only)
            app = []
            for r in self._receipts.get(case_id, []):
                if r.output_hash != c.output_hash:
                    continue
                if r.rule_rev != c.rule_rev:
                    continue
                if r.obs_rev != c.obs_rev:
                    continue
                if r.time < c.t0 or r.time > cutoff:
                    continue
                app.append(r)

            # Qualified rejection check
            for r in app:
                if r.decision == "reject":
                    if r.reviewer not in self._registry:
                        continue
                    if r.roles_at_review & c.roles:
                        return False, "qualified_rejection"

            # Role coverage and distinct count
            covered: Set[str] = set()
            qualified: Set[str] = set()
            for r in app:
                if r.decision != "approve":
                    continue
                if r.reviewer not in self._registry:
                    continue
                q = r.roles_at_review & c.roles
                if q:
                    covered |= q
                    qualified.add(r.reviewer)

            if not c.roles.issubset(covered):
                return False, "missing_role_coverage"
            if len(qualified) < c.min_k:
                return False, "too_few_distinct_reviewers"

        # Permit
        self._released.add(case_id)
        return True, None

    def reset_released(self, case_id: str):
        """For testing: allow re-evaluation."""
        self._released.discard(case_id)
