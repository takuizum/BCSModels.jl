# Changelog

All notable changes to BCSModels.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-15

First public release.

### Added

- **Two model families**:
  - `IRTBCSM`: single-layer probit IRT with a covariance `Σ = I + θ · 1 1ᵀ`
    (Fox 2024). Targets small-sample 2PL applications.
  - `TestletBCSM`: disjoint-testlet probit IRT with one rank-one covariance
    layer per item bundle (Fox, Wenzel & Klotzke 2021).
- **Two posterior inference engines** for each model family, sharing priors
  and Albert–Chib augmentation so they can be compared head-to-head:
  - `gibbs_irt_bcsm` / `gibbs_testlet_bcsm` — closed-form Gibbs samplers
    using the truncated-shifted-inverse-gamma conditional posterior derived
    in Klotzke & Fox (2019a) and the Geweke one-at-a-time scan for latent
    utilities.
  - `cavi_irt_bcsm` / `cavi_testlet_bcsm` — original mean-field
    coordinate-ascent variational Bayes routines with closed-form `q(θ)`
    updates and Albert–Chib truncated-normal `q(z)` marginals. To our
    knowledge this is the first VB inference engine for the BCSM family.
- **`AdditiveCovariance`** struct with Woodbury and Sherman–Morrison
  utilities for the BCSM marginal covariance.
- **`TruncatedShiftedInverseGamma`** distribution with sampling, mean,
  variance, and `E[1/(θ+ψ)]` evaluated through incomplete-gamma integrals.
- **Data utilities**: `simulate_irt_bcsm`, `simulate_testlet_bcsm`, and a
  TIMSS 2019 Grade-8 mathematics loader stub.
- **Diagnostics**: `posterior_summary`, `coverage`, `ess_basic`,
  `vb_summary`.
- **Reproduction scripts**:
  - `scripts/sim_mcmc_vs_vb.jl` — MCMC vs VB comparison grid.
  - `scripts/param_recovery.jl` — N × K parameter recovery study.
  - `scripts/analyze_results.jl`, `scripts/analyze_recovery.jl` — analysers.
  - `scripts/timss_analysis.jl` — placeholder for real-data analysis.
- **Documentation**:
  - `docs/theory.md` — full manuscript-style derivation of the Gibbs and
    CAVI updates, with Sherman–Morrison and conditional-posterior algebra.
  - `docs/derivations.md` — condensed implementation reference.
  - `docs/mcmc_vs_vb.md` — empirical MCMC vs VB comparison
    (17 cells × 20 replicates).
  - `docs/param_recovery.md` — N × K recovery study
    (32 cells × 15 replicates).
  - `paper/paper.md` — JOSS submission draft.
- **Quality tooling**:
  - 17 numerical/recovery tests.
  - Aqua.jl static checks (ambiguities, stale deps, compat, piracy).
  - JET.jl static analysis (`report_package`).
  - CI on Julia 1.10 (LTS) and 1 (stable), Linux and macOS, with Codecov.
  - CompatHelper and TagBot GitHub Actions.

### Known limitations

- Mean-field CAVI systematically underestimates the posterior SD of `θ`
  (median sd-ratio 0.41 vs MCMC) and shows a non-vanishing negative bias
  of `θ` on multi-layer Testlet-BCSM. The bias is structural — the
  variational `q(z) = ∏_{i,j} q(z_{ij})` removes the within-testlet
  correlation that the data carries. A block-structured variational
  family is the planned remedy; see `docs/theory.md` §10.
- TIMSS loader is a placeholder. Users must extend `load_booklet` after
  downloading the SPSS data.
