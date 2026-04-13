"""
scFM — Single-Cell Foundation Model
=====================================
A foundation model for single-cell transcriptomics, built from scratch
with step-by-step commentary on every architectural decision.

Modules
-------
module1_foundations   : Gene vocabulary, value binning, input embeddings
module2_transformer   : Masked attention transformer, GEP pretraining
module3_finetuning    : GEPC, ECS, DAR, cell type classification
module4_training_grn  : Training loop, perturbation prediction, GRN inference
"""

from .module1_foundations import (
    GeneVocab,
    value_binning,
    scFMInputEmbedding,
    scFMDataPreprocessor,
)
from .module2_transformer import (
    scFMModel,
    scFMTransformer,
    scFMTransformerBlock,
    MultiHeadAttention,
    GeneExpressionPredictionHead,
    build_scfm_attention_mask,
    compute_gep_loss,
)
from .module3_finetuning import (
    scFMFineTuner,
    GEPCHead,
    ElasticCellSimilarityLoss,
    DomainAdaptationClassifier,
    CellTypeClassifier,
)
from .module4_training_grn import (
    scFMTrainer,
    scFMTrainingConfig,
    PerturbationPredictor,
    GRNInference,
)

__all__ = [
    # Module 1
    "GeneVocab",
    "value_binning",
    "scFMInputEmbedding",
    "scFMDataPreprocessor",
    # Module 2
    "scFMModel",
    "scFMTransformer",
    "scFMTransformerBlock",
    "MultiHeadAttention",
    "GeneExpressionPredictionHead",
    "build_scfm_attention_mask",
    "compute_gep_loss",
    # Module 3
    "scFMFineTuner",
    "GEPCHead",
    "ElasticCellSimilarityLoss",
    "DomainAdaptationClassifier",
    "CellTypeClassifier",
    # Module 4
    "scFMTrainer",
    "scFMTrainingConfig",
    "PerturbationPredictor",
    "GRNInference",
]
