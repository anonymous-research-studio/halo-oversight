"""
Discrete-event simulator for HALO allocation study.

Poisson arrivals over 5 simulated days of 480 minutes.
Specialist/generalist queues with 30-minute withholding deadline.

Every released case passes through a ReleaseGate: obligated cases carry a
contract requiring a specialist approval, and the release is attempted only
after the review receipt is recorded.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .gate import Contract, ReleaseGate, output_hash
from .policies import BasePolicy


# ── Modality parameters (Table 1) ────────────────────────────────────

MODALITIES = {
    1: {"name": "Passive telemetry", "blocking": False, "effort": 0, "catch_prob": 0.0},
    2: {"name": "Flagged audit", "blocking": False, "effort": 0.5, "catch_prob": 0.0},
    3: {"name": "Live monitoring", "blocking": False, "effort": 1, "catch_prob": 0.50},
    4: {"name": "Component review", "blocking": True, "effort": 3, "catch_prob": 0.80},
    5: {"name": "Full approval", "blocking": True, "effort": 5, "catch_prob": 0.90},
    6: {"name": "Guided co-pilot", "blocking": True, "effort": 8, "catch_prob": 0.95},
    7: {"name": "Human takeover", "blocking": True, "effort": 12, "catch_prob": 0.97},
}
MAX_EFFORT = max(m["effort"] for m in MODALITIES.values())


# ── Hazard types ─────────────────────────────────────────────────────

HAZARD_TYPES = [None, "dosage_error", "contraindication", "wrong_patient",
                "missing_allergy", "outdated_guideline"]
SEVERITY_WEIGHTS = {
    "dosage_error": 5,
    "contraindication": 2,
    "wrong_patient": 5,
    "missing_allergy": 1,
    "outdated_guideline": 2,
}
CRITICAL_SEVERITY = 5

# Three SVL rules. Each fires on an observable proxy (features), never on the
# hazard label, and each also fires on some benign cases. "hazards" names the
# hazard types a rule was written for; it is used only to report which hazard
# types no rule encodes.
#   Rule 1: low confidence + high sensitivity  (dosage_error, contraindication)
#   Rule 2: high user vulnerability             (wrong_patient)
#   Rule 3: more than 3 components + low conf.  (missing_allergy)
# features[7] is component_count / 8, so "more than 3 components" is > 3/8.
RULE_HAZARD_COVERAGE = {
    "rule_1": {"hazards": {"dosage_error", "contraindication"},
               "feature_check": lambda f: f[0] < 0.4 and f[3] > 0.6},
    "rule_2": {"hazards": {"wrong_patient"},
               "feature_check": lambda f: f[4] > 0.7},
    "rule_3": {"hazards": {"missing_allergy"},
               "feature_check": lambda f: f[7] > 3 / 8 and f[0] < 0.5},
}

# Hazard types that no rule was written for, at any coverage level.
UNENCODED_HAZARDS = frozenset(
    set(HAZARD_TYPES[1:]) - set().union(*(r["hazards"] for r in RULE_HAZARD_COVERAGE.values()))
)

SPECIALIST_ROLE = "specialist"
GENERALIST_ROLE = "generalist"


@dataclass
class Case:
    case_id: str
    arrival_time: float
    features: np.ndarray
    hazard_type: Optional[str]
    severity: float
    output_text: str


@dataclass
class SimConfig:
    arrival_rate: float = 1.0        # arrivals per minute
    n_generalists: int = 4
    n_specialists: int = 2
    sim_days: int = 5
    day_length: float = 480.0        # minutes per day
    withhold_deadline: float = 30.0  # minutes
    hazard_prob: float = 0.14        # P(hazardous)
    rule_coverage: float = 1.0       # probability each rule is applied to a case
    cost_weight_effort: float = 0.02
    cost_weight_withheld: float = 1.0
    seed: int = 42


@dataclass
class SimResults:
    total_cases: int = 0
    total_harm: float = 0.0
    total_effort: float = 0.0
    total_withheld: int = 0
    total_released: int = 0
    total_caught: int = 0
    blocking_interventions: int = 0
    any_interventions: int = 0
    hazardous_cases: int = 0
    critical_hazards: int = 0
    critical_caught: int = 0
    true_positives: int = 0       # discretionary interventions on hazardous cases
    false_positives: int = 0      # discretionary interventions on safe cases
    obligated_cases: int = 0
    unencoded_hazard_cases: int = 0
    unencoded_hazards_released: int = 0   # released with harm (not caught, not withheld)
    gate_permits: int = 0
    gate_holds: int = 0
    obligated_by_action: dict = field(default_factory=dict)  # enforced level -> count
    cost_weight_effort: float = 0.02
    cost_weight_withheld: float = 1.0

    def _per_1000(self, value):
        if self.total_cases == 0:
            return 0.0
        return value * 1000.0 / self.total_cases

    @property
    def cost_per_1000(self):
        return self._per_1000(self.total_harm
                              + self.cost_weight_effort * self.total_effort
                              + self.cost_weight_withheld * self.total_withheld)

    @property
    def harm_per_1000(self):
        return self._per_1000(self.total_harm)

    @property
    def effort_per_1000(self):
        return self._per_1000(self.total_effort)

    @property
    def withheld_per_1000(self):
        return self._per_1000(self.total_withheld)

    @property
    def block_pct(self):
        return self._per_1000(self.blocking_interventions) / 10.0

    @property
    def any_pct(self):
        return self._per_1000(self.any_interventions) / 10.0

    @property
    def disc_precision(self):
        """Share of the policy's own (discretionary) interventions that fell
        on hazardous cases; NaN if the policy never intervened on its own."""
        n = self.true_positives + self.false_positives
        if n == 0:
            return float('nan')
        return 100.0 * self.true_positives / n

    @property
    def critical_catch_pct(self):
        if self.critical_hazards == 0:
            return 0.0
        return 100.0 * self.critical_caught / self.critical_hazards

    @property
    def unencoded_released_pct(self):
        if self.unencoded_hazard_cases == 0:
            return float('nan')
        return 100.0 * self.unencoded_hazards_released / self.unencoded_hazard_cases


def generate_case(rng: np.random.RandomState, case_id: str,
                  arrival_time: float, hazard_prob: float) -> Case:
    """Generate a single case with noisy features."""
    if rng.random() < hazard_prob:
        hazard_type = rng.choice(HAZARD_TYPES[1:])
        severity = SEVERITY_WEIGHTS[hazard_type]
    else:
        hazard_type = None
        severity = 0.0

    # Hazardous cases have lower confidence and higher sensitivity on average.
    if hazard_type is not None:
        compound_conf = np.clip(rng.normal(0.35, 0.15), 0, 1)
        mean_comp_conf = np.clip(rng.normal(0.4, 0.15), 0, 1)
        conf_spread = np.clip(rng.normal(0.3, 0.1), 0, 1)
        topic_sens = np.clip(rng.normal(0.7, 0.15), 0, 1)
        vulnerability = np.clip(rng.normal(0.6, 0.2), 0, 1)
        latency = np.clip(rng.normal(0.5, 0.2), 0, 1)
        prior_triggers = rng.poisson(2)
        component_count = rng.randint(2, 8)
    else:
        compound_conf = np.clip(rng.normal(0.75, 0.12), 0, 1)
        mean_comp_conf = np.clip(rng.normal(0.8, 0.1), 0, 1)
        conf_spread = np.clip(rng.normal(0.1, 0.08), 0, 1)
        topic_sens = np.clip(rng.normal(0.3, 0.15), 0, 1)
        vulnerability = np.clip(rng.normal(0.3, 0.15), 0, 1)
        latency = np.clip(rng.normal(0.3, 0.15), 0, 1)
        prior_triggers = rng.poisson(0.5)
        component_count = rng.randint(1, 5)

    features = np.array([
        compound_conf,
        mean_comp_conf,
        conf_spread,
        topic_sens,
        vulnerability,
        latency,
        min(prior_triggers / 5.0, 1.0),  # normalized
        component_count / 8.0,           # normalized
    ])

    output_text = f"output_{case_id}_{rng.randint(0, 1000000)}"

    return Case(case_id, arrival_time, features, hazard_type, severity, output_text)


def check_svl_rules(features: np.ndarray, rule_coverage: float,
                    rng: np.random.RandomState) -> Tuple[int, bool]:
    """Apply the SVL rules to one case. Returns (floor, needs_specialist).

    Rule coverage c means each rule is applied to a case with probability c,
    independently per case and rule. One draw per rule is always consumed,
    so the random stream is identical at every coverage level.
    """
    floor = 1
    needs_specialist = False
    for rule_def in RULE_HAZARD_COVERAGE.values():
        if rng.random() > rule_coverage:
            continue
        if rule_def["feature_check"](features):
            floor = max(floor, 5)  # specialist-approval obligation
            needs_specialist = True
    return floor, needs_specialist


@dataclass
class Workload:
    """Everything random about one run, drawn before any policy acts.

    Every policy run on the same config sees identical cases, rule firings
    and catch draws (common random numbers), so differences between
    policies come only from their decisions.
    """
    cases: list
    floors: list
    needs_specialist: list
    catch_draws: np.ndarray


def generate_workload(config: SimConfig) -> Workload:
    rng = np.random.RandomState(config.seed)
    total_time = config.sim_days * config.day_length

    arrivals = []
    t = 0.0
    while t < total_time:
        t += rng.exponential(1.0 / config.arrival_rate)
        if t < total_time:
            arrivals.append(t)

    cases, floors, needs_specialist = [], [], []
    for idx, arr_time in enumerate(arrivals):
        case = generate_case(rng, f"c{idx:05d}", arr_time, config.hazard_prob)
        floor, spec = check_svl_rules(case.features, config.rule_coverage, rng)
        cases.append(case)
        floors.append(floor)
        needs_specialist.append(spec)
    catch_draws = rng.random_sample(len(cases))
    return Workload(cases, floors, needs_specialist, catch_draws)


def run_simulation(policy: BasePolicy, config: SimConfig) -> SimResults:
    """Run one simulation with the given policy and configuration."""
    workload = generate_workload(config)
    results = SimResults(cost_weight_effort=config.cost_weight_effort,
                         cost_weight_withheld=config.cost_weight_withheld)
    results.total_cases = len(workload.cases)

    # Reviewer pools: time at which each reviewer becomes free.
    gen_free = [0.0] * config.n_generalists
    spec_free = [0.0] * config.n_specialists

    gate = ReleaseGate()
    for i in range(config.n_generalists):
        gate.register_reviewer(f"gen_{i}", {GENERALIST_ROLE})
    for i in range(config.n_specialists):
        gate.register_reviewer(f"spec_{i}", {SPECIALIST_ROLE})
    rule_revision = f"rules@coverage={config.rule_coverage}"

    for idx, case in enumerate(workload.cases):
        arr_time = case.arrival_time
        floor = workload.floors[idx]
        needs_specialist = workload.needs_specialist[idx]
        is_obligated = floor > 1
        if is_obligated:
            results.obligated_cases += 1

        h = output_hash(case.output_text)
        gate.issue(Contract(
            case_id=case.case_id,
            output_hash=h,
            rule_revision=rule_revision,
            obs_revision="o1",
            required_roles=frozenset({SPECIALIST_ROLE}) if needs_specialist else frozenset(),
            min_reviewers=1,
            min_action=floor,
            window_start=arr_time,
            deadline=arr_time + config.withhold_deadline + MAX_EFFORT,
        ))

        queue_len = None
        if policy.sees_queue:
            queue_len = sum(1 for ft in gen_free if ft > arr_time) + \
                        sum(1 for ft in spec_free if ft > arr_time)
        proposed = policy.select(case.features, queue_len)
        enforced = max(proposed, floor)

        if is_obligated:
            results.obligated_by_action[enforced] = results.obligated_by_action.get(enforced, 0) + 1

        modality = MODALITIES[enforced]
        is_blocking = modality["blocking"]
        effort = modality["effort"]
        catch_prob = modality["catch_prob"]

        if is_blocking:
            results.blocking_interventions += 1
        if enforced > 1:
            results.any_interventions += 1

        # Discretionary interventions: the policy intervened on its own,
        # on a case no rule obligated.
        if proposed > 1 and not is_obligated:
            if case.hazard_type is not None:
                results.true_positives += 1
            else:
                results.false_positives += 1

        if case.hazard_type is not None:
            results.hazardous_cases += 1
            if case.severity >= CRITICAL_SEVERITY:
                results.critical_hazards += 1
            if case.hazard_type in UNENCODED_HAZARDS:
                results.unencoded_hazard_cases += 1

        release_time = arr_time
        if is_blocking:
            pool, prefix = (spec_free, "spec") if needs_specialist else (gen_free, "gen")
            earliest = min(pool)
            reviewer_idx = pool.index(earliest)
            start_time = max(arr_time, earliest)
            if start_time - arr_time > config.withhold_deadline:
                # Withheld: never released; prevents harm, costs 1.0.
                results.total_withheld += 1
                policy.update(case.features, enforced, -config.cost_weight_withheld)
                continue
            end_time = start_time + effort
            pool[reviewer_idx] = end_time
            gate.record_review(case.case_id, f"{prefix}_{reviewer_idx}", "approve", end_time)
            release_time = end_time

        results.total_effort += effort

        harm = 0.0
        if case.hazard_type is not None:
            if workload.catch_draws[idx] < catch_prob:
                results.total_caught += 1
                if case.severity >= CRITICAL_SEVERITY:
                    results.critical_caught += 1
            else:
                harm = case.severity
                results.total_harm += harm
                if case.hazard_type in UNENCODED_HAZARDS:
                    results.unencoded_hazards_released += 1

        permitted, _ = gate.release(case.case_id, h, rule_revision, "o1",
                                    release_time, enforced)
        if permitted:
            results.gate_permits += 1
        else:
            results.gate_holds += 1
        results.total_released += 1

        policy.update(case.features, enforced,
                      -(harm + config.cost_weight_effort * effort))

    policy.end_episode()
    return results
