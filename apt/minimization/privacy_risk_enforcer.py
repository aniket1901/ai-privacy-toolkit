"""
Risk-driven privacy enforcement based on dataset membership inference attacks.
"""
from typing import Optional

from apt.risk.data_assessment.dataset_attack_membership_classification import DatasetAttackMembershipClassification, DatasetAttackConfigMembershipClassification
from apt.risk.data_assessment.dataset_attack_membership_knn_probabilities import DatasetAttackMembershipKnnProbabilities, DatasetAttackConfigMembershipKnnProbabilities


class PrivacyRiskEnforcer:
    """
    Wrapper around membership inference attacks used as a risk signal.

    For the membership classification attack, risk_score is the normalized ratio
    (member_auc / non_member_auc - 1). Absolute AUC caps prevent false-safe cases
    where the ratio looks acceptable but the released dataset is still trivially
    distinguishable from both member and non-member data.
    """

    def __init__(self,
        attack_type="membership_classification",
        max_risk=0.05,
        max_member_auc=0.90,
        max_non_member_auc=0.90,
        require_no_warning=True,
        enforcement="auto",
        max_iters=10
    ):
        '''
        :param attack_type: Attack implementation used to compute risk.
                       One of: "membership_classification", "membership_knn_probabilities".
        :type attack_type: str, optional
        :param max_risk: Maximum allowed `risk_score` (>= 0). If None, this check is disabled.
        :type max_risk: float, optional
        :param max_member_auc: Maximum allowed member AUC (0..1). If None, this check is disabled.
        :type max_member_auc: float, optional
        :param max_non_member_auc: Maximum allowed non-member AUC (0..1). If None, this check is disabled.
        :type max_non_member_auc: float, optional
        :param require_no_warning: If True, fail when the attack reports a data-quality warning.
        :type require_no_warning: bool, optional
        :param enforcement: "auto" or "raise". In "raise" mode, violations raise ValueError.
        :type enforcement: str, optional
        :param max_iters: Safety limit for iterative enforcement loops in the minimizer (>= 1).
        :type max_iters: int, optional
        '''
        if attack_type not in ("membership_classification", "membership_knn_probabilities"):
            raise ValueError("attack_type must be membership_classification or membership_knn_probabilities")
        if enforcement not in ("auto", "raise"):
            raise ValueError("enforcement must be auto or raise")
        if max_risk is not None and max_risk < 0:
            raise ValueError("max_risk must be >= 0")
        if max_member_auc is not None and not (0.0 <= max_member_auc <= 1.0):
            raise ValueError("max_member_auc must be in [0,1]")
        if max_non_member_auc is not None and not (0.0 <= max_non_member_auc <= 1.0):
            raise ValueError("max_non_member_auc must be in [0,1]")
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        self.attack_type = attack_type
        self.max_risk = max_risk
        self.max_member_auc = max_member_auc
        self.max_non_member_auc = max_non_member_auc
        self.require_no_warning = require_no_warning
        self.enforcement = enforcement
        self.max_iters = max_iters

    def is_enabled(self):
        ''' 
        Return True if risk enforcement is active.
        '''
        return self.max_risk is not None

    def evaluate(self, member_dataset, non_member_dataset, generalized_dataset) -> dict:
        '''
        Run the configured membership attack and return standardized metrics.

        :param member_dataset: Dataset representing members (training data).
        :param non_member_dataset: Dataset representing non-members (holdout/test data).
        :param generalized_dataset: Released dataset produced by minimization.
        :return: Dict containing keys:
                 - "risk_score" (float)
                 - "member_auc" (float|None)
                 - "non_member_auc" (float|None)
                 - "warning" (bool)
        :rtype: dict
        '''
        if self.attack_type == "membership_classification":
            attack = DatasetAttackMembershipClassification(
                member_dataset,
                non_member_dataset,
                generalized_dataset,
                DatasetAttackConfigMembershipClassification()
            )
            score = attack.assess_privacy()
            return {
                "risk_score": float(score.risk_score),
                "member_auc": float(score.member_roc_auc_score),
                "non_member_auc": float(score.non_member_roc_auc_score),
                "warning": bool(score.synthetic_data_quality_warning)
            }

        attack = DatasetAttackMembershipKnnProbabilities(
            member_dataset,
            non_member_dataset,
            generalized_dataset,
            DatasetAttackConfigMembershipKnnProbabilities()
        )
        score = attack.assess_privacy()
        return {
            "risk_score": float(score.risk_score),
            "member_auc": None,
            "non_member_auc": None,
            "warning": False
        }

    def check(self, metrics_dict):
        '''
        Return True if the provided metrics satisfy all configured thresholds.
        '''
        if not self.is_enabled():
            return True
        if self.max_risk is not None and metrics_dict["risk_score"] > self.max_risk:
            return False
        if self.max_member_auc is not None and metrics_dict["member_auc"] is not None and metrics_dict["member_auc"] > self.max_member_auc:
            return False
        if self.max_non_member_auc is not None and metrics_dict["non_member_auc"] is not None and metrics_dict["non_member_auc"] > self.max_non_member_auc:
            return False
        if self.require_no_warning and metrics_dict["warning"]:
            return False
        return True

    def on_violation(self, metrics_dict):
        '''
        Handles a risk-policy violation based on enforcement mode
        '''
        if not self.is_enabled():
            return
        if self.enforcement == "raise":
            raise ValueError(
                "Privacy risk violated: risk={} max_risk={} member_auc={} max_member_auc={} "
                "non_member_auc={} max_non_member_auc={} warning={} require_no_warning={}".format(
                    metrics_dict["risk_score"], self.max_risk,
                    metrics_dict["member_auc"], self.max_member_auc,
                    metrics_dict["non_member_auc"], self.max_non_member_auc,
                    metrics_dict["warning"], self.require_no_warning
                )
            )
