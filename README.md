# BCSModels.jl

[![CI](https://github.com/takuizum/BCSModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/takuizum/BCSModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/takuizum/BCSModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/takuizum/BCSModels.jl)
[![Julia ≥ 1.10](https://img.shields.io/badge/Julia-%E2%89%A5%201.10-9558B2.svg?logo=julia&logoColor=white)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


**Bayesian Covariance Structure Modeling for Item Response Theory in Julia.**

Julia implementation of Bayesian Covariance Structure Models (BCSM) developed by Fox,
Mulder, Klotzke and colleagues, with a focus on IRT applications. The package provides
both the classical Gibbs sampler used in the literature and an original mean-field
variational Bayes (CAVI) inference routine derived for this project.

## 30-second tour

```julia
using BCSModels, Random, Statistics

# 1. Simulate 500 respondents × 10 items from the single-layer IRT-BCSM
#    with covariance Σ = I + θ_true · 1 1ᵀ.
Y, info = simulate_irt_bcsm(MersenneTwister(1), 500, 10; θ_true=0.4)

# 2. Fit the same model via MCMC (closed-form Gibbs from the BCSM literature)
#    and via mean-field CAVI (this package's original contribution).
model = IRTBCSM(K=10)
gibbs = gibbs_irt_bcsm(Y, model; niter=1500, burnin=500)
vb    = cavi_irt_bcsm(Y, model; maxiter=400, tol=1e-7)

# 3. Compare the posterior of θ.
mean(gibbs.samples_θ)   # ≈ 0.4    (truth)
vb.m_θ[1]               # variational mean (under-disperses, see docs/mcmc_vs_vb.md)
```

Multi-testlet BCSM is one extra constructor:

```julia
testlet_of = repeat(1:3, inner=4)   # 12 items split into 3 testlets of size 4
Y, info = simulate_testlet_bcsm(MersenneTwister(2), 600, 12;
                                 testlet_of=testlet_of,
                                 θ_true=[0.3, 0.5, 0.2])
model = TestletBCSM(K=12, testlet_of=testlet_of)
gibbs_testlet_bcsm(Y, model; niter=1500, burnin=500).samples_θ |> x -> mean(x, dims=1)
```

## Why BCSM in one paragraph

A hierarchical IRT model with random person or testlet effects induces a marginal
covariance among the item-response utilities once the random effects are integrated
out. **BCSM places the prior directly on that marginal covariance:**

> Σ  =  Σ₀  +  ∑ₜ θₜ uₜ uₜᵀ

where each `uₜ` is a known binary item-grouping indicator (e.g. testlet membership)
and each `θₜ ∈ ℝ` is a covariance component. Allowing `θₜ < 0` admits dependence
patterns that the random-effects formulation cannot represent. With a
truncated-shifted-inverse-gamma prior on each `θₜ`, the conditional posteriors are
all closed form, enabling both a fast Gibbs sampler and (this package's
contribution) a closed-form mean-field CAVI. See [`docs/theory.md`](docs/theory.md)
for the full derivation.

## Scope

The implementation targets two main models:

1. **IRT-BCSM** for small-sample 2PL data (Fox, 2024, *JEBS*).
2. **Testlet-BCSM** for tests with item bundles such as TIMSS content/cognitive domains
   (Fox, Wenzel & Klotzke, 2021, *JEBS*).

For each model, the package provides:

- A Gibbs sampler (closed-form conditional posteriors via truncated shifted
  inverse-gamma updates).
- A mean-field coordinate-ascent variational inference (CAVI) routine, derived
  using Albert–Chib probit augmentation.
- Posterior summary, model fit, and diagnostic utilities.

## Layout

```
src/
  BCSModels.jl                 # package entrypoint
  distributions/          # truncated shifted inverse-gamma, helpers
  covariance/             # additive covariance struct + rank-1 updates
  models/
    irt_bcsm.jl           # Fox (2024) IRT-BCSM definition
    testlet_bcsm.jl       # Testlet-BCSM definition
  mcmc/
    gibbs_irt.jl          # Gibbs sampler for IRT-BCSM
    gibbs_testlet.jl      # Gibbs sampler for Testlet-BCSM
  vb/
    cavi_irt.jl           # mean-field CAVI for IRT-BCSM
    cavi_testlet.jl       # mean-field CAVI for Testlet-BCSM
  data/
    simulation.jl         # simulate BCSM responses
    timss.jl              # TIMSS 2019 grade-8 math loader
  diagnostics.jl
test/
scripts/
  sim_mcmc_vs_vb.jl       # simulation experiment
  timss_analysis.jl       # real data analysis
docs/
  theory.md               # full manuscript-style derivation (for paper / users)
  derivations.md          # condensed implementation note
  mcmc_vs_vb.md           # MCMC vs CAVI empirical comparison
  param_recovery.md       # parameter-recovery experiment across N × K
```

## Installation

Once registered in the Julia General registry:

```julia
using Pkg
Pkg.add("BCSModels")
```

Until then (or to develop the package locally):

```julia
using Pkg
Pkg.add(url="https://github.com/takuizum/BCSModels.jl")          # from GitHub
# or, for a local checkout:
Pkg.develop(path="/path/to/BCSModels.jl")
```

## Getting started

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'    # 17 tests, ~10 s
```

Example: simulate, fit, compare:

```julia
using BCSModels, Random
Y, info = simulate_irt_bcsm(MersenneTwister(1), 500, 10; θ_true=0.4)
model   = IRTBCSM(K=10)
gibbs   = gibbs_irt_bcsm(Y, model; niter=1500, burnin=500)
vb      = cavi_irt_bcsm(Y, model; maxiter=400, tol=1e-7)
```

Full simulation grid (≈ 20 reps, several minutes):

```bash
julia --project=. scripts/sim_mcmc_vs_vb.jl --reps 20
```

TIMSS 2019 Grade-8 mathematics (requires manual data download — see
`scripts/download_timss.sh`):

```bash
bash scripts/download_timss.sh                      # prints instructions
julia --project=. scripts/timss_analysis.jl --country USA --booklet 1
```

## Empirical findings

### MCMC vs. VB on the full grid — 17 cells × 20 replicates

Full details and the per-cell table live in [`docs/mcmc_vs_vb.md`](docs/mcmc_vs_vb.md).
Aggregate across cells:

| Metric                          | MCMC      | VB        |
|---------------------------------|-----------|-----------|
| mean \|bias\|                    | **0.014** | **0.117** |
| mean RMSE                       | **0.061** | **0.128** |
| posterior SD ratio (VB / MCMC) median | — | **0.41** |
| 95% CI coverage                 | **0.94**  | **0.18**  |
| speedup (MCMC time / VB time) median | — | **11.5×** |

The summary: **MCMC is the reference; mean-field CAVI matches it only for
the simple single-layer IRT-BCSM at modest θ.** For multi-layer testlet
settings VB shows a systematic 30 – 45 % *negative* bias on the covariance
components, and the variational 95 % credible interval covers the truth
only ~18 % of the time. VB is 10–20× faster, so it is useful as a
point-estimate screening tool, but not as a stand-alone inference engine.

The bias source is structural: the mean-field factorisation
`q(z) = ∏_{i,j} q(z_{ij})` removes the within-testlet correlation between
latent utilities that carries the signal for `θ_t`. A block-structured
variational family `q(z_i) = N_K(μ_i, S_i)` is the obvious next step and
would likely close most of the gap.

### Parameter recovery across `N × K` — 32 cells × 15 replicates

A more systematic recovery study (see
[`docs/param_recovery.md`](docs/param_recovery.md)) holds three findings:

- **Item difficulties** `b` are recovered nearly identically by both
  engines (correlation with truth ≥ 0.984 in every cell; RMSE differs by
  < 0.02 absolute). The VB credible interval for `b` is slightly too tight
  (coverage 76 % vs 95 % nominal).
- **MCMC on `θ`** is consistent: `|bias|` drops from 0.085 at `N=100, K=10`
  to 0.002 at `N=1000, K=20`, and 95 % coverage stays at the nominal 0.93.
- **VB on `θ`** has a *non-vanishing* downward bias that does not shrink
  with `N`. Its posterior SD shrinks even faster, so 95 % CI coverage
  **drops** as `N` grows: 0.90 at `N=100` → 0.00–0.30 at `N=1000, K=10`.
  Coverage on the Testlet-BCSM with `N ≥ 500` is below 0.20 — unusable
  for inference, fine for point-estimate screening.

VB is 13× faster on average. The pattern is consistent with the
structural-bias analysis in [`docs/theory.md`](docs/theory.md) §7.7.

## Theoretical documentation

A self-contained, manuscript-style derivation of the model, the Gibbs
sampler, and the new mean-field CAVI is in
[`docs/theory.md`](docs/theory.md). It is intended both as the basis for a
Psychometrika / JEBS submission and as theoretical support for users of
this package. Highlights:

- the explicit algebraic identity (Proposition 5.1) that makes
  $\psi_t = 1/k_t$ the *unique* shift under which the conditional
  posterior of every covariance component is a pure inverse-gamma;
- a step-by-step Gibbs derivation, with all matrix identities shown;
- a full CAVI derivation under conditional conjugacy, including the ELBO
  and the plug-in approximation $\mathbb{E}_q[\Lambda] \approx
  \Lambda(\mathbb{E}_q[\theta])$;
- a structural explanation of why mean-field VB under-estimates
  $\theta_t$ in multi-layer testlet settings (§7.7), and the natural
  next-step structured-VB upgrade that would close the gap;
- pseudocode for both algorithms (Appendix A).

## Key references

**BCSM core (Fox & collaborators).** Curated from Fox's
[publications overview](https://www.jean-paulfox.com/publications-overview/).

- Mulder, J., & Fox, J.-P. (2013). Bayesian tests on components of the compound
  symmetry covariance matrix. *Statistics and Computing*, 23(1), 109–122.
- Fox, J.-P., Mulder, J., & Sinharay, S. (2017). Bayes factor covariance testing
  in item response models. *Psychometrika*, 82(4), 979–1006.
- Mulder, J., & Fox, J.-P. (2019). Bayes factor testing of multiple intraclass
  correlations. *Bayesian Analysis*, 14(2), 521–552.
- Klotzke, K., & Fox, J.-P. (2019a). Bayesian covariance structure modeling of
  responses and process data. *Frontiers in Psychology*, 10:1675.
- Klotzke, K., & Fox, J.-P. (2019b). Modeling dependence structures for response
  times in a Bayesian framework. *Psychometrika*, 84(3), 649–672.
- Fox, J.-P., Wenzel, J., & Klotzke, K. (2021). The Bayesian covariance structure
  model for testlets. *JEBS*, 46(2), 219–243.
- Fox, J.-P., Koops, J., Feskens, R., & Beinhauer, L. (2020). Bayesian
  covariance structure modelling for measurement invariance testing.
  *Behaviormetrika*, 47, 385–410.
- Fox, J.-P., & Smink, W. A. C. (2021). Assessing an alternative for "negative
  variance components": A gentle introduction to BCSM for negative
  associations. *arXiv:2106.10107*.
- Nielsen, N. M., Smink, W. A. C., & Fox, J.-P. (2021). Small and negative
  correlations among clustered observations: limitations of the linear
  mixed-effects model. *Behaviormetrika*, 48, 51–77.
- Baas, S., Boucherie, R. J., & Fox, J.-P. (2022). Bayesian covariance
  structure modeling of multi-way nested data. *arXiv:2201.10612*.
- Baas, S., Boucherie, R. J., & Fox, J.-P. (2024). Bayesian covariance
  structure modeling of interval-censored multi-way nested survival data.
  *Journal of Multivariate Analysis*, 204.
- Fox, J.-P. (2024). Redefining item response models for small samples.
  *JEBS*, 50(2), 272–295.
- Fox, J.-P. (2026, in press). Bayesian covariance modeling of differential
  item functioning. *Psychometrika*. doi:10.1017/psy.2026.10101.

**Foundations and broader context.**

- Fox, J.-P. (2010). *Bayesian Item Response Modeling: Theory and
  Applications*. Springer. (textbook covering the multilevel-IRT Gibbs
  sampler that BCSM marginalises.)
- Fox, J.-P., van den Berg, S. M., & Veldkamp, B. P. (2018). Bayesian
  psychometric methods. In *Handbook of Psychometric Testing* (Wiley).
- Fox, J., Klotzke, K., & Simsek, A. S. (2023). R-package LNIRT for joint
  modeling of response accuracy and times. *PeerJ Computer Science*,
  9:e1232.
- Mulder, J., et al. (2019). BFpack: Flexible Bayes factor testing of
  scientific theories in R. *arXiv:1911.07728*.

**Variational inference for IRT (used in our CAVI derivation / comparison).**

- Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for
  logistic models using Pólya–Gamma latent variables. *JASA*, 108(504),
  1339–1349.
- Cho, A. E., Wang, C., Zhang, X., & Xu, G. (2021). Gaussian variational
  estimation for multidimensional IRT. *BJMSP*, 74, 52–85.
- Wand, M. P. (2017). Fast approximate inference for arbitrarily large
  semiparametric regression models via message passing. *JASA*, 112(517),
  137–168.
- Maestrini, L., & Wand, M. P. (2021). The inverse G-Wishart distribution
  and variational message passing. *ANZJStat*, 63, 517–541.
