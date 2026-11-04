"""Model contracts, decision policy, and auditable evaluation."""

from parcel_sorter.ml.binary_evaluation import (
    BinaryCandidateValidationResult,
    BinaryDecisionCode,
    BinaryEvaluationReport,
    BinaryPredictionRecord,
    BinaryThresholds,
    BinaryThresholdSelection,
    apply_binary_policy,
    evaluate_binary_predictions,
    load_binary_prediction_table,
    save_binary_prediction_table,
    select_binary_model_candidate,
    select_binary_validation_thresholds,
)
from parcel_sorter.ml.contracts import (
    AugmentationContract,
    DecisionThresholds,
    ExperimentConfig,
    ModelCandidate,
    PreprocessContract,
)
from parcel_sorter.ml.evaluation import (
    ClassMetric,
    EvaluationReport,
    PredictionRecord,
    ThresholdSelection,
    ThresholdSweepPoint,
    evaluate_predictions,
    select_validation_thresholds,
)
from parcel_sorter.ml.policy import (
    DecisionCode,
    DecisionResult,
    apply_decision_policy,
    is_confirmed_damage,
)

__all__ = [
    "AugmentationContract",
    "BinaryCandidateValidationResult",
    "BinaryDecisionCode",
    "BinaryEvaluationReport",
    "BinaryPredictionRecord",
    "BinaryThresholdSelection",
    "BinaryThresholds",
    "ClassMetric",
    "DecisionCode",
    "DecisionResult",
    "DecisionThresholds",
    "EvaluationReport",
    "ExperimentConfig",
    "ModelCandidate",
    "PredictionRecord",
    "PreprocessContract",
    "ThresholdSelection",
    "ThresholdSweepPoint",
    "apply_binary_policy",
    "apply_decision_policy",
    "evaluate_binary_predictions",
    "evaluate_predictions",
    "is_confirmed_damage",
    "load_binary_prediction_table",
    "save_binary_prediction_table",
    "select_binary_model_candidate",
    "select_binary_validation_thresholds",
    "select_validation_thresholds",
]
