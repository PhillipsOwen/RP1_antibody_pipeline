# helx Project Memory

## Last Session — 2026-02-24: RP1 Gap Closure

Implemented all five gaps identified in a spec-vs-implementation review:

| Gap | What was done |
|-----|---------------|
| 1 (high) | Added `MMPBSACalculator` to `binding_md.py`; rewrote `_md_interface_energy()` to use OpenMM vacuum minimisation (AMBER14) with sigmoid ΔE→score; auto-fallback to embedding proxy |
| 2 (high) | Created `md_simulations/structural_pathways.py` (`AgAbComplexBuilder`, `BindingPathwaySimulator`); added `stage_structural_pathways()` to `main.py` (stage 2.5, between 2b and 2c) |
| 3 (med)  | Added `save_checkpoint()` / `load_checkpoint()` to `AntibodyVAE`; `stage_structure()` now saves to `MODELS_DIR/vae_checkpoint.pt` and auto-loads on re-run |
| 4 (med)  | Expanded `MDConfig` docstring (RCSB, AlphaFold DB, ESMFold, PDBFixer); added data-source `logger.info` in `stage_msm()` when MD files are absent (non-mock) |
| 5 (low)  | Expanded `BCRConfig` docstring (OAS URL + download steps); added OAS guidance `logger.info` in `stage_bcr_repertoire()` when OAS dir missing (non-mock) |

Verified: `python -m RP1_antibody_pipeline.main --mock` runs clean end-to-end (~5 s).
Files changed: `config.py`, `main.py`, `models/vae.py`, `md_simulations/binding_md.py`,
`md_simulations/structural_pathways.py` (new).

---


## Active Research: RP1 — Antibody Responses to Viral Escape Mutants

### RP1_antibody_pipeline/ — structure
```
RP1_antibody_pipeline/
  config.py                          # PipelineConfig + all sub-configs
  main.py                            # pipeline orchestrator (run_pipeline)
  requirements.txt                   # pip deps + annotations for GROMACS/CHARMM
  data/
    SARS-CoV-2_sequences.fasta       # 10 complete SARS-CoV-2 genomes (2026, USA)
    get-virus-fasta.py               # NCBI Entrez download script
    bcr_loader.py                    # OASLoader, PrivateBCRLoader, load_repertoire()
    oas/                             # (user-provided) OAS bulk-download CSVs
  models/
    antibody_lm.py                   # ESM2 LM wrapper + build_atlas() + atlas_similarity()
    vae.py                           # VAE + save_checkpoint() + load_checkpoint()
    gan.py                           # GAN for structural realism
    alm_finetuner.py                 # ALMFineTuner — MD-guided pairwise ranking loss
  md_simulations/
    md_runner.py                     # OpenMMRunner + GROMACSRunner + CHARMMRunner + get_md_runner()
    binding_md.py                    # BindingMDPredictor + MMPBSACalculator — physics or proxy ΔG
    structural_pathways.py           # AgAbComplexBuilder, BindingPathwaySimulator
  msm_analysis/msm_builder.py        # MSMBuilder (PyEMMA or NumPy fallback)
  synthetic_evolution/evolution.py   # RepertoireEvolver, CDR-focused mutation
  viral_escape/
    escape_mutant.py                 # EscapeMutantGenerator, EscapeMutant
    binding_predictor.py             # CrossReactivityScorer
    antigen_profile.py               # AntigenBindingSiteProfiler — ALM epitope vs CDR
    blind_spot.py                    # BlindSpotAnalyzer — immune blind spot detection
  experiments/
    validation.py                    # ValidationDataset, generate_escape_report
    lab_loop.py                      # LabInTheLoop — iterative lab refinement
  utils/helpers.py                   # parallel_map, load_spike_from_fasta (biopython+ORF scan),
                                     # load_all_spike_sequences_from_fasta
```

### Pipeline stages (main.py run_pipeline)
```
0.   stage_escape_panel          — viral escape mutant panel from wildtype antigen
1.   stage_bcr_repertoire        — OAS + private BCR loading; build disease atlas
2.   stage_lm                    — antibody LM scoring/generation (ESM2)
2a.  stage_antigen_alm_profile   — spike variants vs ALM binding sites
2b.  stage_md_binding_prediction — embedding-proxy or MM/PBSA MD binding scores
2.5. stage_structural_pathways   — mock Ag-Ab complexes + pathway MSM
2c.  stage_alm_finetune          — fine-tune LM FFN layers on MD binding scores
2d.  stage_blind_spot_analysis   — immune blind spots: atlas vs antigen coverage
3.   stage_structure             — VAE + GAN conformational modelling
4.   stage_msm                   — MD trajectory loading + conformational MSM
5.   stage_evolution             — escape-aware affinity maturation
6.   stage_repertoire_screen     — parallel LM scoring, top candidates
7.   stage_cross_reactivity      — CrossReactivityScorer, coverage matrix
8.   stage_vaccine_design        — broadly neutralising candidate selection
9.   stage_validation            — experimental comparison + escape coverage report
10.  stage_lab_loop              — ingest wet-lab results, re-fine-tune, active learning
```

### Key design decisions
- BCR atlas: ESM2 mean-pool embeddings over OAS repertoire → centroid + std
- Antigen profiling: epitope-weighted embedding similarity (cosine default)
- MD binding proxy: L2 distance on unit sphere → [0,1] binding score
- MMPBSACalculator: ΔE=E_complex−E_receptor−E_ligand via OpenMM vacuum minimisation
  (AMBER14 in-vacuo); sigmoid 1/(1+exp(ΔE/kT)) → [0,1]; auto-fallback to embedding
  proxy when OpenMM absent or no PDB provided; pdb_source kwarg on BindingMDPredictor
- ALM fine-tuning: pairwise MarginRankingLoss on PLLs; only FFN layers updated
- MD backends: openmm (default) | gromacs | charmm — set in MDConfig.backend
- VAE checkpoint: save_checkpoint(path) / AntibodyVAE.load_checkpoint(path, device)
  stored at MODELS_DIR/vae_checkpoint.pt; auto-loaded in stage_structure (non-mock)
- Structural pathways: AgAbComplexBuilder (mock dict or real MDTraj PDB features) +
  BindingPathwaySimulator (3-phase Gaussian approach/bound/separation trajectory or
  OpenMM steered MD); stage 2.5 builds separate pathway-level MSM
- Data guidance logs: OAS URL logged in stage_bcr_repertoire when dir missing (non-mock);
  RCSB/AlphaFold/PDBFixer URLs logged in stage_msm when MD files absent (non-mock)
- FASTA spike extraction: biopython SeqIO → reference coords → ORF scan fallback
- ViralEscapeConfig.antigen_sequence loaded via load_spike_from_fasta()
- epitope_residues: known ACE2-contact sites in spike RBD (0-indexed)

### Config fields
- BCRConfig: oas_data_dir, private_data_path, disease_label, max_sequences, atlas_output_path
  (docstring has full OAS bulk-download instructions)
- MDConfig: backend, pdb_source, forcefield, topology_file, trajectory_file, temperature_k,
  simulation_steps, step_size_fs, report_interval
  (docstring has RCSB PDB / AlphaFold DB / ESMFold / PDBFixer acquisition steps)
- AntigenALMConfig: n_antigen_sequences, similarity_metric
- ALMFinetuneConfig: learning_rate, n_epochs, margin, min_binding_score

### Run commands
```bash
# Project venv python
.venv/Scripts/python.exe -m RP1_antibody_pipeline.main --mock   # fast dry-run (~5 s)
.venv/Scripts/python.exe -m RP1_antibody_pipeline.main          # full run (GPU + data)
```

### Requirements
numpy, scipy, scikit-learn, torch, transformers (ESM2), mdtraj, matplotlib, seaborn,
pandas, biopython
External (conda): openmm, gromacs, charmm (academic licence), pyemma (optional), ray (optional)
BCR data:      https://opig.stats.ox.ac.uk/webapps/oas/
PDB structures: https://www.rcsb.org/   |   https://alphafold.ebi.ac.uk/
Structure prep: pip install pdbfixer
