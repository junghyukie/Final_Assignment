"""COVER-EVO research implementation.

Importing this package is deliberately offline and has no provider side effects.
"""

from .contracts import (
    ArchiveEntry,
    EvalResult,
    ExperimentConfig,
    FeatureVector,
    Instance,
    ModelConfig,
    OracleResult,
    Program,
    ProviderResponse,
    ProposalRecord,
    SelectionManifest,
    SolveResult,
    RunState,
)

__all__ = [
    "ArchiveEntry",
    "EvalResult",
    "ExperimentConfig",
    "FeatureVector",
    "Instance",
    "ModelConfig",
    "OracleResult",
    "Program",
    "ProviderResponse",
    "ProposalRecord",
    "SelectionManifest",
    "SolveResult",
    "RunState",
]

__version__ = "0.1.0"
