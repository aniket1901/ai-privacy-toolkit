import pytest

from apt.minimization.dp_mechanism import DifferentialPrivacyMechanism
from apt.minimization.homogeneity_guard import HomogeneityGuard
from apt.minimization.minimizer import GeneralizeToRepresentative
from apt.minimization.privacy_floor import PrivacyFloorEnforcer
from apt.minimization.privacy_risk_enforcer import PrivacyRiskEnforcer

"""
Unit tests for the minimization privacy controls.

Covers the four added mechanisms: 
  -  DP threshold privatization
  -  NCP privacy floor
  -  Homogeneity guard
  -  Risk-driven membership-inference enforcement
Also checks that GeneralizeToRepresentative set_params/get_params correctly wires these wrappers.
"""

def test_dp_mechanism_validation_and_truncation():
    with pytest.raises(ValueError):
        DifferentialPrivacyMechanism(epsilon=0)

    with pytest.raises(ValueError):
        DifferentialPrivacyMechanism(epsilon=1.0, sensitivity=0)

    with pytest.raises(ValueError):
        DifferentialPrivacyMechanism(epsilon=1.0, delta=-0.1)

    mechanism = DifferentialPrivacyMechanism(epsilon=1.0, random_state=7)

    with pytest.raises(ValueError):
        mechanism.privatize_value(5.0, lower=0.0)

    with pytest.raises(ValueError):
        mechanism.privatize_value(5.0, lower=1.0, upper=1.0)

    with pytest.raises(ValueError):
        mechanism.privatize_value(5.0, sensitivity=0)

    privatized = mechanism.privatize_value(5.0, lower=0.0, upper=10.0)

    assert 0.0 <= privatized <= 10.0


def test_privacy_floor_enforcer_relative_and_absolute_floor():
    enforcer = PrivacyFloorEnforcer(alpha=0.5)
    assert enforcer.is_enabled()

    enforcer.set_baseline(8.0)
    assert enforcer.get_effective_min_ncp() == 4.0
    assert enforcer.check(4.0)
    assert not enforcer.check(3.9)

    absolute = PrivacyFloorEnforcer(min_ncp=6.0, alpha=0.5)
    absolute.set_baseline(20.0)
    assert absolute.get_effective_min_ncp() == 6.0
    assert absolute.check(6.0)
    assert not absolute.check(5.9)


def test_privacy_floor_enforcer_raise_mode():
    enforcer = PrivacyFloorEnforcer(min_ncp=2.5, enforcement="raise")

    with pytest.raises(ValueError, match="NCP privacy floor violated"):
        enforcer.on_violation(2.0)


def test_homogeneity_guard_stats_and_checks():
    guard = HomogeneityGuard(min_cell_size=2, min_label_diversity=2, min_entropy=0.5)

    stats = guard.cell_stats(labels=[0, 0, 1, 1], indices=[0, 2])
    assert stats["size"] == 2
    assert stats["distinct"] == 2
    assert stats["entropy"] > 0.5
    assert guard.check_cell(stats)

    small_stats = guard.cell_stats(labels=[0, 0, 0], indices=[1])
    assert not guard.check_cell(small_stats)

    same_label_stats = guard.cell_stats(labels=[1, 1, 1], indices=[0, 1])
    assert not guard.check_cell(same_label_stats)


def test_homogeneity_guard_raise_mode():
    guard = HomogeneityGuard(min_cell_size=3, enforcement="raise")

    with pytest.raises(ValueError, match="HomogeneityGuard violated"):
        guard.on_violation(cell_id=7, stats={"size": 2, "distinct": 1, "entropy": 0.0})


def test_privacy_risk_enforcer_check_and_raise():
    enforcer = PrivacyRiskEnforcer(
        max_risk=0.05,
        max_member_auc=0.8,
        max_non_member_auc=0.8,
        require_no_warning=True,
        enforcement="raise",
    )

    good_metrics = {
        "risk_score": 0.01,
        "member_auc": 0.7,
        "non_member_auc": 0.7,
        "warning": False,
    }
    bad_metrics = {
        "risk_score": 0.06,
        "member_auc": 0.9,
        "non_member_auc": 0.85,
        "warning": True,
    }

    assert enforcer.check(good_metrics)
    assert not enforcer.check(bad_metrics)

    with pytest.raises(ValueError, match="Privacy risk violated"):
        enforcer.on_violation(bad_metrics)


def test_privacy_risk_enforcer_evaluate_knn_mode(monkeypatch):
    class DummyScore:
        risk_score = 0.123

    class DummyAttack:
        def __init__(self, member_dataset, non_member_dataset, generalized_dataset, config):
            self.member_dataset = member_dataset
            self.non_member_dataset = non_member_dataset
            self.generalized_dataset = generalized_dataset
            self.config = config

        @staticmethod
        def assess_privacy():
            return DummyScore()

    monkeypatch.setattr(
        "apt.minimization.privacy_risk_enforcer.DatasetAttackMembershipKnnProbabilities",
        DummyAttack,
    )

    enforcer = PrivacyRiskEnforcer(attack_type="membership_knn_probabilities", max_risk=0.2)
    metrics = enforcer.evaluate(member_dataset="member", non_member_dataset="non_member", generalized_dataset="gen")

    assert metrics == {
        "risk_score": 0.123,
        "member_auc": None,
        "non_member_auc": None,
        "warning": False,
    }


def test_minimizer_privacy_controls_roundtrip_params():
    gen = GeneralizeToRepresentative()

    gen.set_params(
        epsilon=1.0,
        delta=1e-5,
        dp_random_state=3,
        min_ncp=0.4,
        privacy_enforcement="raise",
        privacy_floor_alpha=0.5,
        homogeneity_min_cell_size=2,
        homogeneity_min_label_diversity=2,
        homogeneity_min_entropy=0.1,
        homogeneity_enforcement="warn",
        max_homogeneity_prune_level=4,
        homogeneity_accuracy_tolerance=0.02,
        risk_max_risk=0.03,
        risk_max_member_auc=0.75,
        risk_max_non_member_auc=0.76,
        risk_require_no_warning=False,
        risk_attack_type="membership_knn_probabilities",
        risk_enforcement="raise",
        max_risk_prune_level=5,
        risk_accuracy_tolerance=0.01,
    )

    params = gen.get_params()

    assert isinstance(gen._dp_mechanism, DifferentialPrivacyMechanism)
    assert isinstance(gen._privacy_floor_enforcer, PrivacyFloorEnforcer)
    assert isinstance(gen._homogeneity_guard, HomogeneityGuard)
    assert isinstance(gen._privacy_risk_enforcer, PrivacyRiskEnforcer)
    assert params["epsilon"] == 1.0
    assert params["min_ncp"] == 0.4
    assert params["privacy_floor_alpha"] == 0.5
    assert params["homogeneity_min_cell_size"] == 2
    assert params["homogeneity_enforcement"] == "warn"
    assert params["risk_max_risk"] == 0.03
    assert params["risk_attack_type"] == "membership_knn_probabilities"
    assert gen._privacy_risk_enforcer.max_iters == 5
