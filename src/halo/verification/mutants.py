"""
16 mutants of the release method, each removing or weakening one check.
2 are equivalent under the current interface (receipts built from stored contract),
so 14 are live.
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..gate import (
    CaseRecord, Contract, HoldReason, Receipt, ReleaseGate, output_hash,
)


class MutantGate(ReleaseGate):
    """Base class for mutant gates. Subclasses override release()."""
    pass


# ── Helper: standard role-check block used by most mutants ───────────
# Each mutant copies the full release() but removes/changes one check.
# The role-check block matches the base gate exactly:
#   - Registry check: if r.reviewer_id not in self._registry: continue
#   - Role intersection: roles_at_review & contract.required_roles


class M01_SkipUnknownCase(MutantGate):
    """Skip the unknown-case check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                # MUTANT: treat unknown as empty record
                record = CaseRecord(Contract(case_id, current_output_hash,
                    current_rule_revision, current_obs_revision,
                    frozenset(), 1, 1, 0.0, float('inf')))
                self._cases[case_id] = record
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M02_SkipAlreadyReleased(MutantGate):
    """Skip the already-released check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            # MUTANT: skip already_released check
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M03_SkipHashCheck(MutantGate):
    """Skip output hash check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            # MUTANT: skip hash check
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M04_SkipRuleRevCheck(MutantGate):
    """Skip rule revision check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            # MUTANT: skip rule revision check
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M05_SkipObsRevCheck(MutantGate):
    """Skip observation revision check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            # MUTANT: skip obs revision check
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M06_SkipWindowCheck(MutantGate):
    """Skip release window check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            # MUTANT: skip window check
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M07_SkipFloorCheck(MutantGate):
    """Skip action floor check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            # MUTANT: skip floor check
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M08_SkipRejectionCheck(MutantGate):
    """Skip qualified rejection check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                # MUTANT: skip rejection check entirely
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M09_SkipRoleCoverage(MutantGate):
    """Skip role coverage check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                # MUTANT: skip role coverage — go straight to distinct count
                approvals = [r for r in applicable if r.decision == "approve"]
                qualified = set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        qualified.add(r.reviewer_id)
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M10_SkipDistinctCount(MutantGate):
    """Skip distinct reviewer count check."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered = set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                # MUTANT: skip distinct count check
            record.released = True
            record.release_time = current_time
            return True, None


class M11_SkipRegistryCheck(MutantGate):
    """Skip checking if reviewer is still registered at release time."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        # MUTANT: skip registry check for rejections
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    # MUTANT: skip registry check for approvals
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M12_SkipReceiptHashCheck(MutantGate):
    """Skip receipt hash matching in applicable check.
    EQUIVALENT under current interface — receipts are built from contract.
    """
    def _applicable_receipts(self, record, contract, current_time):
        cutoff = min(current_time, contract.deadline)
        applicable = []
        for r in record.receipts:
            if r.case_id != contract.case_id:
                continue
            # MUTANT: skip hash check on receipt
            if r.rule_revision != contract.rule_revision:
                continue
            if r.obs_revision != contract.obs_revision:
                continue
            if r.timestamp < contract.window_start or r.timestamp > cutoff:
                continue
            applicable.append(r)
        return applicable


class M13_SkipReceiptRevisionCheck(MutantGate):
    """Skip receipt revision matching in applicable check.
    EQUIVALENT under current interface — receipts are built from contract.
    """
    def _applicable_receipts(self, record, contract, current_time):
        cutoff = min(current_time, contract.deadline)
        applicable = []
        for r in record.receipts:
            if r.case_id != contract.case_id:
                continue
            if r.output_hash != contract.output_hash:
                continue
            # MUTANT: skip revision checks on receipt
            if r.timestamp < contract.window_start or r.timestamp > cutoff:
                continue
            applicable.append(r)
        return applicable


class M14_SkipReceiptTimeCheck(MutantGate):
    """Skip receipt timestamp check in applicable filter."""
    def _applicable_receipts(self, record, contract, current_time):
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
            # MUTANT: skip time check
            applicable.append(r)
        return applicable


class M15_WeakenWindowExclDeadline(MutantGate):
    """Use strict < instead of <= for deadline (excludes deadline instant)."""
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            # MUTANT: strict < instead of <=
            if current_time < contract.window_start or current_time >= contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    q = r.roles_at_review & contract.required_roles
                    if q:
                        covered |= q
                        qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


class M16_SkipCurrentRoleCheck(MutantGate):
    """Skip checking reviewer's roles against required roles for approvals.

    Any registered approval counts as covering all required roles,
    regardless of what roles the reviewer actually held at review time.
    """
    def release(self, case_id, current_output_hash, current_rule_revision,
                current_obs_revision, current_time, enforced_action):
        with self._lock:
            if case_id not in self._cases:
                return False, HoldReason.unknown_case
            record = self._cases[case_id]
            contract = record.contract
            if record.released:
                return False, HoldReason.already_released
            if current_output_hash != contract.output_hash:
                return False, HoldReason.payload_changed
            if current_rule_revision != contract.rule_revision:
                return False, HoldReason.rules_changed
            if current_obs_revision != contract.obs_revision:
                return False, HoldReason.observations_changed
            if current_time < contract.window_start or current_time > contract.deadline:
                return False, HoldReason.outside_release_window
            if enforced_action < contract.min_action:
                return False, HoldReason.action_below_floor
            if contract.required_roles:
                applicable = self._applicable_receipts(record, contract, current_time)
                for r in applicable:
                    if r.decision == "reject":
                        if r.reviewer_id not in self._registry:
                            continue
                        if r.roles_at_review & contract.required_roles:
                            return False, HoldReason.qualified_rejection
                approvals = [r for r in applicable if r.decision == "approve"]
                covered, qualified = set(), set()
                for r in approvals:
                    if r.reviewer_id not in self._registry:
                        continue
                    # MUTANT: skip role intersection — any registered approval
                    # covers all required roles
                    covered |= contract.required_roles
                    qualified.add(r.reviewer_id)
                if not contract.required_roles.issubset(covered):
                    return False, HoldReason.missing_role_coverage
                if len(qualified) < contract.min_reviewers:
                    return False, HoldReason.too_few_distinct_reviewers
            record.released = True
            record.release_time = current_time
            return True, None


# All 16 mutants. M12 and M13 are equivalent (marked).
MUTANTS = [
    ("M01_SkipUnknownCase", M01_SkipUnknownCase, True),
    ("M02_SkipAlreadyReleased", M02_SkipAlreadyReleased, True),
    ("M03_SkipHashCheck", M03_SkipHashCheck, True),
    ("M04_SkipRuleRevCheck", M04_SkipRuleRevCheck, True),
    ("M05_SkipObsRevCheck", M05_SkipObsRevCheck, True),
    ("M06_SkipWindowCheck", M06_SkipWindowCheck, True),
    ("M07_SkipFloorCheck", M07_SkipFloorCheck, True),
    ("M08_SkipRejectionCheck", M08_SkipRejectionCheck, True),
    ("M09_SkipRoleCoverage", M09_SkipRoleCoverage, True),
    ("M10_SkipDistinctCount", M10_SkipDistinctCount, True),
    ("M11_SkipRegistryCheck", M11_SkipRegistryCheck, True),
    ("M12_SkipReceiptHashCheck", M12_SkipReceiptHashCheck, False),  # equivalent
    ("M13_SkipReceiptRevisionCheck", M13_SkipReceiptRevisionCheck, False),  # equivalent
    ("M14_SkipReceiptTimeCheck", M14_SkipReceiptTimeCheck, True),
    ("M15_WeakenWindowExclDeadline", M15_WeakenWindowExclDeadline, True),
    ("M16_SkipCurrentRoleCheck", M16_SkipCurrentRoleCheck, True),
]

assert len(MUTANTS) == 16
assert sum(1 for _, _, live in MUTANTS if live) == 14
