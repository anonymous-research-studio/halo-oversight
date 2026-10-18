"""
Agent adapter that places the release gate in front of any generate(prompt) callable.

Operations: issue a contract, accept a review, regenerate if the pipeline reruns,
release whatever output is current.
"""

import time
from typing import Callable, Optional, Set, FrozenSet

from .gate import Contract, ReleaseGate, output_hash


class AgentAdapter:
    """Wraps a generate(prompt) callable with the obligation-preserving gate.

    Usage:
        def my_agent(prompt: str) -> str:
            return llm.generate(prompt)

        adapter = AgentAdapter(my_agent, gate)
        adapter.register_reviewer("dr_smith", {"pharmacist"})

        result = adapter.run(
            prompt="Adjust metformin dose based on latest labs",
            required_roles=frozenset({"pharmacist"}),
            min_reviewers=1,
            floor=5,
            deadline_seconds=1800,
        )
        # result is None until approved and released
        adapter.submit_review("dr_smith", "approve")
        released = adapter.try_release()
    """

    def __init__(self, generate_fn: Callable[[str], str], gate: Optional[ReleaseGate] = None):
        self._generate = generate_fn
        self._gate = gate or ReleaseGate()
        self._case_counter = 0
        self._current_case: Optional[str] = None
        self._current_output: Optional[str] = None
        self._current_hash: Optional[str] = None
        self._rule_revision = "v1"
        self._obs_revision = "o1"
        self._contract: Optional[Contract] = None
        self.last_hold_reason = None

    @property
    def gate(self) -> ReleaseGate:
        return self._gate

    def register_reviewer(self, reviewer_id: str, roles: Set[str]):
        self._gate.register_reviewer(reviewer_id, roles)

    def revoke_reviewer(self, reviewer_id: str):
        self._gate.revoke_reviewer(reviewer_id)

    def set_rule_revision(self, rev: str):
        self._rule_revision = rev

    def set_obs_revision(self, rev: str):
        self._obs_revision = rev

    def run(
        self,
        prompt: str,
        required_roles: FrozenSet[str] = frozenset(),
        min_reviewers: int = 1,
        floor: int = 1,
        deadline_seconds: float = 1800.0,
    ) -> str:
        """Generate output and issue a contract for it.

        Returns the generated output text. The output is NOT released
        until try_release() succeeds.
        """
        self._case_counter += 1
        case_id = f"adapter_case_{self._case_counter:06d}"

        # Generate output
        output = self._generate(prompt)

        # Compute hash
        h = output_hash(output)

        # Create contract
        now = time.time()
        contract = Contract(
            case_id=case_id,
            output_hash=h,
            rule_revision=self._rule_revision,
            obs_revision=self._obs_revision,
            required_roles=required_roles,
            min_reviewers=min_reviewers,
            min_action=floor,
            window_start=now,
            deadline=now + deadline_seconds,
        )

        # If there was a previous case, revoke it
        if self._current_case is not None:
            self._gate.revoke(self._current_case)

        self._gate.issue(contract)
        self._current_case = case_id
        self._current_output = output
        self._current_hash = h
        self._contract = contract

        return output

    def regenerate(self, prompt: str) -> str:
        """Regenerate output (e.g., after context changes).

        Revokes the old contract and issues a new one (run() does the revoke).
        """
        return self.run(
            prompt,
            required_roles=self._contract.required_roles if self._contract else frozenset(),
            min_reviewers=self._contract.min_reviewers if self._contract else 1,
            floor=self._contract.min_action if self._contract else 1,
            deadline_seconds=(
                (self._contract.deadline - self._contract.window_start)
                if self._contract else 1800.0
            ),
        )

    def submit_review(self, reviewer_id: str, decision: str) -> bool:
        """Submit a review for the current case."""
        if self._current_case is None:
            return False
        receipt = self._gate.record_review(
            self._current_case, reviewer_id, decision, time.time()
        )
        return receipt is not None

    def try_release(self, enforced_action: int = 5,
                    current_time: Optional[float] = None) -> Optional[str]:
        """Attempt to release the current output.

        Returns the output text if released, None if held. ``current_time``
        defaults to the wall clock; pass a value for simulation or testing.
        """
        if self._current_case is None or self._current_hash is None:
            return None
        t = current_time if current_time is not None else time.time()
        permitted, reason = self._gate.release(
            self._current_case,
            self._current_hash,
            self._rule_revision,
            self._obs_revision,
            t,
            enforced_action,
        )
        self.last_hold_reason = None if permitted else reason
        if not permitted:
            return None
        output = self._current_output
        self._current_case = None
        self._current_output = None
        self._current_hash = None
        self._contract = None
        return output

    def try_release_sim(self, enforced_action: int = 5,
                        current_time: Optional[float] = None) -> Optional[str]:
        """Alias of try_release, kept for compatibility."""
        return self.try_release(enforced_action, current_time)
