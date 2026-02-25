"""
viral_escape — Viral escape mutant generation and cross-reactivity scoring.

Modules
-------
escape_mutant     : EscapeMutant dataclass + EscapeMutantGenerator
binding_predictor : CrossReactivityScorer + coverage matrix building
"""

from .escape_mutant import EscapeMutant, EscapeMutantGenerator
from .binding_predictor import CrossReactivityScorer

__all__ = ["EscapeMutant", "EscapeMutantGenerator", "CrossReactivityScorer"]