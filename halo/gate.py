"""
Obligation-preserving release gate for HALO.

A single Python module exposing four operations: issue, record_review, revoke, release.
Outputs are bound by SHA-256 over their UTF-8 encoding.
167 lines of logic (excluding docstrings/comments), no dependencies outside stdlib.
"""

import hashlib
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


class HoldReason(Enum):
    unknown_case = "unknown_case"
    already_released = "already_released"
    payload_changed = "payload_changed"
    rules_changed = "rules_changed"
    observations_changed = "observations_changed"
    outside_release_window = "outside_release_window"
    action_below_floor = "action_below_floor"
    missing_role_coverage = "missing_role_coverage"
    qualified_rejection = "qualified_rejection"
    too_few_distinct_reviewers = "too_few_distinct_reviewers"


@dataclass(frozen=True)
class Contract:
    case_id: str
    output_hash: str          # SHA-256 hex digest of the output
    rule_revision: str        # identifies the rule set version
    obs_revision: str         # identifies the observation set version
    required_roles: FrozenSet[str]  # O_i: set of roles whose approval is required
    min_reviewers: int        # k_i >= 1: minimum distinct approvers when roles non-empty
    min_action: int           # a_i^min: oversight floor (1-7)
    window_start: float       # t_i^0: start of release window
    deadline: float           # d_i: end of release window


@dataclass(frozen=True)
class Receipt:
    case_id: str
    reviewer_id: str
    roles_at_review: FrozenSet[str]  # roles the reviewer held when reviewing
    output_hash: str
    rule_revision: str
    obs_revision: str
    decision: str             # "approve" or "reject"
    timestamp: float


@dataclass
class CaseRecord:
    contract: Contract
    receipts: List[Receipt] = field(default_factory=list)
    released: bool = False
    release_time: Optional[float] = None


def output_hash(text: str) -> str:
    """SHA-256 over UTF-8 encoding of the output text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReleaseGate:
    """Obligation-preserving release gate.

    Four operations: issue, record_review, revoke, release.
    """

    def __init__(self):
        self._cases: Dict[str, CaseRecord] = {}
        self._registry: Dict[str, Set[str]] = {}  # reviewer_id -> current roles
        self._lock = threading.Lock()

    # ── Registry management ──────────────────────────────────────────

    def register_reviewer(self, reviewer_id: str, roles: Set[str]) -> None:
        self._registry[reviewer_id] = set(roles)

    def revoke_reviewer(self, reviewer_id: str) -> None:
        """Remove a reviewer from the registry (revoke)."""
        self._registry.pop(reviewer_id, None)

    def get_reviewer_roles(self, reviewer_id: str) -> FrozenSet[str]:
        return frozenset(self._registry.get(reviewer_id, set()))

    # ── Four operations ──────────────────────────────────────────────

    def issue(self, contract: Contract) -> None:
        """Issue a contract for a case."""
        self._cases[contract.case_id] = CaseRecord(contract=contract)

    def record_review(
        self,
        case_id: str,
        reviewer_id: str,
        decision: str,
        timestamp: float,
    ) -> Optional[Receipt]:
        """Record a review. Roles are taken from the registry, not the caller."""
        if case_id not in self._cases:
            return None
        record = self._cases[case_id]
        contract = record.contract
        roles_now = self.get_reviewer_roles(reviewer_id)
        receipt = Receipt(
            case_id=case_id,
            reviewer_id=reviewer_id,
            roles_at_review=roles_now,
            output_hash=contract.output_hash,
            rule_revision=contract.rule_revision,
            obs_revision=contract.obs_revision,
            decision=decision,
            timestamp=timestamp,
        )
        record.receipts.append(receipt)
        return receipt

    def revoke(self, case_id: str) -> bool:
        """Revoke a contract (e.g. when output is regenerated)."""
        if case_id not in self._cases:
            return False
        # Remove the case entirely so it must be re-issued
        del self._cases[case_id]
        return True

    def release(
        self,
        case_id: str,
        current_output_hash: str,
        current_rule_revision: str,
        current_obs_revision: str,
        current_time: float,
        enforced_action: int,
    ) -> Tuple[bool, Optional[HoldReason]]:
        """Check release conditions and, if all pass, record the release.

        Returns (True, None) on permit, (False, reason) on hold.
        Checks are evaluated in sequence; first failure returns.
        """
        with self._lock:
            # 1. Case is known
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract

            # 2. Not already released
            if record.released:
                return False, HoldReason.already_released

            # 3. Output hash matches
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed

            # 4. Rule revision matches
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed

            # 5. Observation revision matches
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed

            # 6. Within release window: t in [t_i^0, d_i]
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window

            # 7. Enforced action meets floor
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor

            # 8-10 only apply when roles are required
            if contract.required_roles:
                # Collect applicable receipts (filtered by hash, revisions, time)
                applicable = self._applicable_receipts(
                    record, contract, current_time
                )

                # 8. Check for qualified rejection (blocks release)
                for r in applicable:
                    if r.decision == "reject":
                        # Reviewer must still be registered
                        if r.reviewer_id not in self._registry:
                            continue
                        # Rejection qualifies if reviewer held a required role at review time
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection

                # 9. Role coverage: every required role covered by an applicable approval
                approvals = [r for r in applicable if r.decision == "approve"]
                covered_roles: Set[str] = set()
                qualified_approvers: Set[str] = set()
                for r in approvals:
                    # Reviewer must still be registered
                    if r.reviewer_id not in self._registry:
                        continue
                    # Role counts if reviewer held it when reviewing
                    qualifying = r.roles_at_review & contract.required_roles
                    if qualifying:
                        covered_roles |= qualifying
                        qualified_approvers.add(r.reviewer_id)

                if not contract.required_roles.issubset(covered_roles):
                    return False, HoldReason.missing_role_coverage

                # 10. Distinct reviewer count
                if len(qualified_approvers) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers

            # All checks passed — record release
            record.released = True
            record.release_time = current_time
            return True, None

    def _applicable_receipts(
        self, record: CaseRecord, contract: Contract, current_time: float
    ) -> List[Receipt]:
        """Filter receipts to those applicable at release time t.

        A receipt is applicable if:
        - belongs to this case
        - carries the contract's hash and revisions
        - was completed within [t_i^0, min(t, d_i)]

        Registration and current-role checks are in release(), not here,
        so they can be independently verified by mutation testing.
        """
        cutoff = min(current_time, contract.deadline)
        applicable = []
        for r in record.receipts:
            if r.case_id != contract.case_id:
                continue
            if r.output_hash != contract.output_hash:
                continue
            if r.rule_revision != contract.rule_revision:
                continue
            if r.obs_revision != contract.obs_revision:
                continue
            if r.timestamp < contract.window_start or r.timestamp > cutoff:
                continue
            applicable.append(r)
        return applicable

    def get_case(self, case_id: str) -> Optional[CaseRecord]:
        return self._cases.get(case_id)
