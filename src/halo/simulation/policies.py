"""
Eight allocation policies for the HALO discrete-event simulator.

Each policy receives 8 noisy features and proposes an action 1-7.
None observes the queue except the capacity-aware threshold (labeled).

Learning policies (LinUCB bandit, linear Q-learner) update from per-case
rewards until ``freeze()`` is called; after that, ``update()`` is a no-op
and action selection is deterministic.
"""

from typing import Optional

import numpy as np


class BasePolicy:
    """Base class for all allocation policies."""
    name: str = "base"
    sees_queue: bool = False
    learns: bool = False

    def __init__(self):
        self.frozen = False

    def select(self, features: np.ndarray, queue_len: Optional[int] = None) -> int:
        raise NotImplementedError

    def update(self, features: np.ndarray, action: int, reward: float):
        """Optional: update the policy from feedback."""
        pass

    def end_episode(self):
        """Called once at the end of each simulation run."""
        pass

    def freeze(self):
        """Stop learning. Selection must be deterministic afterwards."""
        self.frozen = True


class MinObligation(BasePolicy):
    """Always propose level 1 (passive telemetry)."""
    name = "Min. obligation"

    def select(self, features, queue_len=None):
        return 1


class StaticGates(BasePolicy):
    """Static gates on a fixed risk score (B2).

    Risk score = 1 - compound_confidence + topic_sensitivity.
    If risk > threshold, propose level 5 (full approval).
    """
    name = "Static gates (B2)"

    def __init__(self, threshold=0.8):
        super().__init__()
        self.threshold = threshold

    def select(self, features, queue_len=None):
        # features[0] = compound_confidence, features[3] = topic_sensitivity
        risk_score = 1.0 - features[0] + features[3]
        if risk_score > self.threshold:
            return 5
        return 1


class ConfidenceThreshold(BasePolicy):
    """Tuned confidence threshold (B3).

    If compound_confidence < threshold, propose level 5.
    """
    name = "Conf. threshold (B3)"

    def __init__(self, threshold=0.6):
        super().__init__()
        self.threshold = threshold

    def select(self, features, queue_len=None):
        if features[0] < self.threshold:
            return 5
        return 1


class RiskMatrix(BasePolicy):
    """Hand-built risk matrix (B4).

    Combines confidence, sensitivity, and vulnerability into
    a risk level and maps to an action.
    """
    name = "Risk matrix (B4)"

    def select(self, features, queue_len=None):
        confidence = features[0]
        sensitivity = features[3]
        vulnerability = features[4]

        risk = (1.0 - confidence) * 0.4 + sensitivity * 0.3 + vulnerability * 0.3

        if risk > 0.7:
            return 5  # full approval
        elif risk > 0.5:
            return 3  # live monitoring
        elif risk > 0.3:
            return 2  # flagged audit
        return 1  # passive


class CapacityAwareThreshold(BasePolicy):
    """Confidence threshold that lowers its cutoff as reviewers get busy.

    The one policy that sees reviewer load: ``queue_len`` is the number of
    reviewers busy at the case's arrival time (see simulator.run_simulation).
    """
    name = "Cap.-aware threshold"
    sees_queue = True

    def __init__(self, base_threshold=0.6, queue_sensitivity=0.05):
        super().__init__()
        self.base_threshold = base_threshold
        self.queue_sensitivity = queue_sensitivity

    def select(self, features, queue_len=None):
        q = queue_len if queue_len is not None else 0
        threshold = self.base_threshold - self.queue_sensitivity * q
        threshold = max(threshold, 0.2)
        if features[0] < threshold:
            return 5
        return 1


class TwoThreshold(BasePolicy):
    """Two-threshold policy (post hoc comparator).

    Monitoring below one confidence cutoff, full approval below a lower one.
    """
    name = "Two-threshold (post hoc)"

    def __init__(self, monitor_threshold=0.7, approval_threshold=0.4):
        super().__init__()
        self.monitor_threshold = monitor_threshold
        self.approval_threshold = approval_threshold

    def select(self, features, queue_len=None):
        confidence = features[0]
        if confidence < self.approval_threshold:
            return 5  # full approval
        elif confidence < self.monitor_threshold:
            return 3  # live monitoring
        return 1


class LinUCBBandit(BasePolicy):
    """LinUCB contextual bandit (B5).

    Standard LinUCB with per-action weight vectors and confidence bounds.
    Deterministic: ties go to the lowest action index.
    """
    name = "LinUCB bandit (B5)"
    learns = True

    def __init__(self, n_features=8, n_actions=7, alpha=1.0):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.A = [np.eye(self.n_features) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_actions)]

    def select(self, features, queue_len=None):
        x = features[:self.n_features]
        best_action = 1
        best_ucb = -float('inf')
        for a_idx in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a_idx])
            theta = A_inv @ self.b[a_idx]
            ucb = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a_idx + 1  # actions are 1-indexed
        return best_action

    def update(self, features, action, reward):
        if self.frozen:
            return
        x = features[:self.n_features]
        a_idx = action - 1
        self.A[a_idx] += np.outer(x, x)
        self.b[a_idx] += reward * x


class LinearQLearner(BasePolicy):
    """Reference linear Q-learner.

    8 features coarse-coded into 16 bins each, linear action values.
    903 parameters (8 x 16 x 7 weights + 7 biases), discount 0.95.

    Transitions run from one case to the next arriving case: the TD target
    for case t bootstraps on the features of case t+1. Exploration uses a
    private, seeded random generator so runs are reproducible.
    """
    name = "Ref. linear Q-learner"
    learns = True

    def __init__(self, n_features=8, n_bins=16, n_actions=7,
                 gamma=0.95, lr=0.01, epsilon=0.1, seed=0):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.feature_dim = n_features * n_bins  # 128
        self.rng = np.random.RandomState(seed)
        self.weights = np.zeros((self.n_actions, self.feature_dim))
        self.bias = np.zeros(self.n_actions)
        self._last_phi = None   # features of the case most recently selected
        self._pending = None    # (phi, action, reward) awaiting next state

    @property
    def n_params(self):
        return self.weights.size + self.bias.size

    def _coarse_code(self, features: np.ndarray) -> np.ndarray:
        """Coarse-code 8 features into 128-dim sparse vector.

        Each feature discretized into 16 bins; primary bin at 1.0,
        adjacent bins at 0.5.
        """
        phi = np.zeros(self.feature_dim)
        for i in range(self.n_features):
            val = np.clip(features[i], 0.0, 1.0)
            bin_idx = int(val * (self.n_bins - 1))
            bin_idx = min(bin_idx, self.n_bins - 1)
            offset = i * self.n_bins
            phi[offset + bin_idx] = 1.0
            if bin_idx > 0:
                phi[offset + bin_idx - 1] = 0.5
            if bin_idx < self.n_bins - 1:
                phi[offset + bin_idx + 1] = 0.5
        return phi

    def _q_values(self, phi: np.ndarray) -> np.ndarray:
        return self.weights @ phi + self.bias

    def _td_update(self, phi, action, reward, phi_next):
        a_idx = action - 1
        target = reward
        if phi_next is not None:
            target += self.gamma * np.max(self._q_values(phi_next))
        error = target - (self.weights[a_idx] @ phi + self.bias[a_idx])
        self.weights[a_idx] += self.lr * error * phi
        self.bias[a_idx] += self.lr * error

    def select(self, features, queue_len=None):
        phi = self._coarse_code(features[:self.n_features])
        if self._pending is not None:
            prev_phi, prev_action, prev_reward = self._pending
            self._td_update(prev_phi, prev_action, prev_reward, phi)
            self._pending = None
        if not self.frozen and self.rng.random_sample() < self.epsilon:
            action = int(self.rng.randint(1, self.n_actions + 1))
        else:
            action = int(np.argmax(self._q_values(phi))) + 1
        self._last_phi = phi
        return action

    def update(self, features, action, reward):
        if self.frozen or self._last_phi is None:
            return
        # Completed on the next select(), when the next state is known.
        self._pending = (self._last_phi, action, reward)

    def end_episode(self):
        if self._pending is not None and not self.frozen:
            prev_phi, prev_action, prev_reward = self._pending
            self._td_update(prev_phi, prev_action, prev_reward, None)
        self._pending = None
        self._last_phi = None

    def freeze(self):
        super().freeze()
        self._pending = None
        self.epsilon = 0.0


# Tuning configurations for each tunable policy (at most eight each).
# Keys are the policy keys used in halo.allocation.
TUNING_CONFIGS = {
    "static_gates": [{"threshold": t} for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]],
    "conf_threshold": [{"threshold": t} for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]],
    "cap_aware": [
        {"base_threshold": bt, "queue_sensitivity": qs}
        for bt in [0.5, 0.6, 0.7, 0.8]
        for qs in [0.03, 0.05]
    ],
    "two_threshold": [
        {"monitor_threshold": mt, "approval_threshold": at}
        for mt in [0.6, 0.7, 0.8]
        for at in [0.3, 0.4, 0.5]
    ][:8],
    "linucb": [{"alpha": a} for a in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]],
    "q_learner": [
        {"lr": lr, "epsilon": eps}
        for lr in [0.001, 0.005, 0.01, 0.05]
        for eps in [0.05, 0.1]
    ],
}
