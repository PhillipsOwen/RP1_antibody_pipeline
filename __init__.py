"""
RP1_antibody_pipeline
=====================
RP1: Predicting Antibody Responses to Viral Escape Mutants
Computational antibody discovery pipeline.

Modules
-------
config              : global configuration dataclasses
models.antibody_lm  : ESM2-based antibody language model
models.vae          : Variational Autoencoder for conformation space
models.gan          : Generative Adversarial Network for structure generation
md_simulations      : MDTraj / OpenMM MD runners and feature extractors
msm_analysis        : Markov State Model construction (PyEMMA or NumPy)
synthetic_evolution : Repertoire evolution / affinity maturation simulation
viral_escape        : Escape mutant generation and cross-reactivity scoring
experiments         : Experimental validation metrics and plots
utils.helpers       : Parallel evaluation, I/O, embeddings

Quick start
-----------
    from RP1_antibody_pipeline.main import run_pipeline
    results = run_pipeline(mock=True)
"""

__version__ = "0.1.0"
