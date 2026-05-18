---
title: 'BCSModels.jl: A Julia package for Bayesian covariance structure modeling of item response data'
tags:
  - Julia
  - Bayesian inference
  - item response theory
  - variational Bayes
  - psychometrics
  - covariance structure modeling
authors:
  - name: Takumi Itamiya
    orcid: 0000-0000-0000-0000   # TODO: replace with real ORCID
    affiliation: 1
affiliations:
  - name: Dwango Co., Ltd.
    index: 1
date: 15 May 2026
bibliography: paper.bib
---

# Summary

`BCSModels.jl` is a Julia package for fitting Bayesian Covariance Structure
Models (BCSM) to binary item-response data. BCSM, developed by Fox, Mulder,
Klotzke and collaborators [@klotzke_fox_2019_responses; @fox2024;
@fox_wenzel_klotzke_2021], reformulates the marginal covariance of
multilevel and testlet item-response-theory (IRT) models as an additive
sum of rank-one layers, $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_0 + \sum_{t=1}^{T} \theta_t \mathbf{u}_t \mathbf{u}_t^{\!\top}$,
where each $\mathbf{u}_t$ is a known item-grouping indicator and each
$\theta_t \in \mathbb{R}$ is a covariance component that — unlike a
variance — is allowed to be negative. Closed-form conditional posteriors
make a Gibbs sampler efficient; the package implements that sampler and,
in addition, a closed-form mean-field coordinate-ascent variational Bayes
(CAVI) routine that, to our knowledge, has not previously been derived
for the BCSM family.

The package targets two model families: the single-layer IRT-BCSM for
small-sample 2PL probit data [@fox2024], and the disjoint-testlet BCSM
[@fox_wenzel_klotzke_2021] with one rank-one layer per item bundle. The
two inference engines share a common probit / Albert–Chib augmentation
[@albert_chib_1993] and the same prior structure, so they can be
compared back-to-back on identical simulated and real data sets. The
package ships with a parameter-recovery script that exercises both
engines across a grid of sample sizes and test lengths and reports
recovery of both the covariance components $\theta_t$ and the item
difficulties $\mathbf{b}$.

# Statement of need

Despite a decade of methodological development, BCSM has no Julia
implementation. The reference code, accompanying Fox and collaborators'
publications, is in R and centred on Gibbs samplers
[@klotzke_fox_2019_responses; @fox_klotzke_simsek_2023_lnirt]. Within the
broader Julia ecosystem there are general-purpose probabilistic-programming
frameworks (`Turing.jl`, `AdvancedVI.jl`) but no model-specific package
that exploits the closed-form structure of BCSM. Two practical gaps
follow:

1. **Performance.** A generic MCMC backend pays significant overhead for
   the latent-utility scan and the rank-one covariance updates that BCSM
   admits in closed form. Hand-written Gibbs samplers in the literature
   are 10–100× faster than equivalent Turing.jl / Stan implementations.

2. **No variational alternative.** All existing BCSM implementations
   are MCMC-based. Variational Bayes is a natural fit because the same
   conditional conjugacy that gives the Gibbs sampler its closed form
   also yields closed-form CAVI updates. Cho et al. [-@cho_wang_zhang_xu_2021]
   derived a Gaussian variational EM for multidimensional IRT, but their
   variational family does not exploit the additive-covariance structure
   that defines BCSM.

`BCSModels.jl` fills both gaps. The package provides the canonical Gibbs
sampler with the algebraic simplifications afforded by the disjoint
binary indicator $\mathbf{u}_t$ — the rank-one layer is an eigenvector
of the rest of the covariance with eigenvalue one, which collapses the
truncation bound to $\mathrm{tr}_t = -1/k_t$ and makes the conditional
posterior of every $\theta_t$ a pure inverse-gamma. The mean-field
CAVI is built on the same algebra and inherits the same closed form for
$q(\theta_t)$.

# Implementation and validation

The package's two model types — `IRTBCSM` and `TestletBCSM` — accept a
binary $N \times K$ response matrix and emit either a chain of posterior
samples (`gibbs_irt_bcsm` / `gibbs_testlet_bcsm`) or a variational
posterior summary (`cavi_irt_bcsm` / `cavi_testlet_bcsm`). The Gibbs
sampler uses Geweke's one-at-a-time truncated-normal scan
[@geweke_1991_efficient] for the latent utilities, a conjugate Gaussian
update for the item difficulties, and the closed-form
truncated-shifted-inverse-gamma update for each $\theta_t$. The CAVI
uses Albert–Chib truncated-normal $q(Z_{ij})$, Gaussian $q(b_j)$, and
inverse-gamma $q(\theta_t)$ marginals; the only non-exact step is the
plug-in $\mathbb{E}_q[\boldsymbol{\Lambda}] \approx \boldsymbol{\Sigma}^{-1}(\mathbb{E}_q[\boldsymbol{\theta}])$
used in the $q(Z)$ and $q(b)$ updates, an approximation in the spirit of
@wand_2017_factor_graph.

A test suite covers (i) sampling from and moment evaluation of the
truncated-shifted-inverse-gamma family, (ii) the Sherman–Morrison form
of the additive covariance, and (iii) parameter recovery of both $\theta$
and $\mathbf{b}$ on simulated data, with the 95% credible interval
covering the truth in at least the nominal proportion of replicates.
A parameter-recovery experiment across $N \in \{100, 250, 500, 1000\}$
and $K \in \{10, 20, 30\}$ shows that the Gibbs sampler is consistent
(bias of $\theta$ shrinks to 0.002 at $N = 1000$, $K = 20$ with 95 %
credible interval coverage at the nominal level), while the mean-field
CAVI carries a non-vanishing $\approx -0.1$ negative bias in $\theta$ due
to the variational family's omission of the within-testlet correlation
between latent utilities. The CAVI runs about 13× faster than the Gibbs
sampler.

# Acknowledgements

This package builds directly on the methodological work of Jean-Paul Fox,
Joris Mulder, and Konrad Klotzke. We thank the Fox lab for their
generosity in publishing reference R implementations alongside their
papers.

# References
